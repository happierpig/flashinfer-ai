"""
Pytest for testing BatchAttentionWithPerHeadSelectPagedKVCacheWrapper against golden reference
"""

import torch
import numpy as np
import pytest
import flashinfer

def generate_test_data(
    batch_size, num_kv_heads, num_qo_heads, head_dim, 
    device, dtype, seed, sparsity_ratio=0.5,
    min_seq_len=100, max_seq_len=500,
    min_q_len=1, max_q_len=50, num_layers=1
):
    """Generate test data for per-head KV selection tests"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate sequence lengths with configurable ranges
    seq_lens = torch.randint(min_seq_len, max_seq_len, (batch_size,), device=device, dtype=torch.int32)
    q_lens = torch.randint(min_q_len, max_q_len, (batch_size,), device=device, dtype=torch.int32)
    
    # Build indptr arrays
    qo_indptr = torch.cat([torch.tensor([0], device=device), torch.cumsum(q_lens, dim=0)], dim=0).int()
    total_query_tokens = qo_indptr[-1].item()
    
    # Determine tokens to select per batch
    selected_tokens_per_batch = []
    for batch_idx in range(batch_size):
        seq_len = seq_lens[batch_idx].item()
        num_selected = max(1, int(seq_len * sparsity_ratio))
        selected_tokens_per_batch.append(num_selected)
    
    kv_indptr = torch.cat([
        torch.tensor([0], device=device),
        torch.cumsum(torch.tensor(selected_tokens_per_batch, device=device), dim=0)
    ], dim=0).int()
    
    # Generate per-head KV indices with shuffling
    all_layers_per_head_kv_indices = []
    for layer_idx in range(num_layers):
        per_head_kv_indices = []
        for head_idx in range(num_kv_heads):
            head_indices = []
            for batch_idx in range(batch_size):
                seq_len = seq_lens[batch_idx].item()
                num_selected = selected_tokens_per_batch[batch_idx]
                
                # Select different random tokens for each head
                selected_local = torch.randperm(seq_len, device=device)[:num_selected].sort()[0]
                
                # Global indices
                global_offset = sum(seq_lens[:batch_idx].tolist()) if batch_idx > 0 else 0
                selected_global = selected_local + global_offset
                
                head_indices.append(selected_global)
            
            head_indices = torch.cat(head_indices)
            per_head_kv_indices.append(head_indices)
        
        per_head_kv_indices = torch.stack(per_head_kv_indices)
        all_layers_per_head_kv_indices.append(per_head_kv_indices)
    
    all_layers_per_head_kv_indices = torch.stack(all_layers_per_head_kv_indices)
    
    # Generate tensors
    query = torch.rand(total_query_tokens, num_qo_heads, head_dim, dtype=dtype, device=device)
    
    # Use original (non-sparsified) total tokens for KV cache size
    original_total_kv_tokens = seq_lens.sum().item()
    page_block_size = 1
    key = torch.rand(original_total_kv_tokens, page_block_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    value = torch.rand(original_total_kv_tokens, page_block_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    
    # Compute last_page_len
    last_page_len = (seq_lens - 1) % page_block_size + 1
    last_page_len = last_page_len.unsqueeze(1).expand(batch_size, num_kv_heads)
    
    sparsified_seq_lens = torch.tensor(selected_tokens_per_batch, device=device, dtype=torch.int32)
    
    return {
        'query': query,
        'key': key,
        'value': value,
        'qo_indptr': qo_indptr,
        'kv_indptr': kv_indptr,
        'all_layers_per_head_kv_indices': all_layers_per_head_kv_indices,
        'seq_lens': seq_lens,
        'sparsified_seq_lens': sparsified_seq_lens,
        'last_page_len': last_page_len,
        'page_block_size': page_block_size,
    }


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
@pytest.mark.parametrize("num_kv_heads,num_qo_heads", [
    (4, 32),  # 1:8 GQA ratio
    (8, 32),  # 1:4 GQA ratio
])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("max_seq_len,max_q_len", [
    (500, 50),    # Default
    (2048, 128),  # Medium
    (8192, 256),  # Large
])
def test_golden_reference(
    batch_size, num_kv_heads, num_qo_heads, head_dim, 
    causal, dtype, max_seq_len, max_q_len, device="cuda"
):
    """Test wrapper against golden reference by running each KV head separately"""
    seed = 42
    
    # Set min lengths relative to max lengths
    min_seq_len = max(100, max_seq_len // 10)
    min_q_len = max(1, max_q_len // 10)
    
    data = generate_test_data(
        batch_size, num_kv_heads, num_qo_heads, head_dim,
        device, dtype, seed,
        min_seq_len=min_seq_len, max_seq_len=max_seq_len,
        min_q_len=min_q_len, max_q_len=max_q_len
    )
    
    # Run BatchAttentionWithPerHeadSelectPagedKVCacheWrapper
    batch_attention_wrapper = flashinfer.BatchAttentionWithPerHeadSelectPagedKVCacheWrapper(device=device)
    layer_idx_buffer = torch.tensor([0], device=device)
    batch_attention_wrapper.plan(
        qo_indptr=data['qo_indptr'],
        kv_indptr=data['kv_indptr'],
        all_layers_per_head_kv_indices=data['all_layers_per_head_kv_indices'].int(), # (num_layers, num_kv_heads, total_kv_indices)
        seq_lens=data['sparsified_seq_lens'],
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        layer_idx=layer_idx_buffer, # a 0-D tensors
        page_block_size=data['page_block_size'],
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
        use_triton=False,
        add_layer_idx_by_one_after_run=True
    )
    output_wrapper, lse_wrapper = batch_attention_wrapper.run(
        data['query'], (data['key'], data['value'])
    )
    # check whether layer_idx_buffer is updated
    assert layer_idx_buffer.item() == 1, "layer_idx_buffer should be updated to 1"
    torch.cuda.synchronize()
    
    # Compute golden reference
    gqa_group_size = num_qo_heads // num_kv_heads
    golden_outputs = []
    golden_lses = []

    for kv_head_idx in range(num_kv_heads):
        # Extract queries for this KV head group
        query_start = kv_head_idx * gqa_group_size
        query_end = (kv_head_idx + 1) * gqa_group_size
        query_per_head = data['query'][:, query_start:query_end, :].contiguous()
        
        # Extract KV cache for this head
        head_kv_indices = data['all_layers_per_head_kv_indices'][0, kv_head_idx] # assuming that num_layers = 1
        key_per_head = data['key'][:, :, kv_head_idx:kv_head_idx+1, :].contiguous()
        value_per_head = data['value'][:, :, kv_head_idx:kv_head_idx+1, :].contiguous()
        
        # Run standard FlashInfer BatchAttention
        wrapper = flashinfer.BatchAttention(kv_layout="NHD")
        wrapper.plan(
            data['qo_indptr'],
            data['kv_indptr'],
            head_kv_indices.int(),
            data['sparsified_seq_lens'],
            gqa_group_size,
            1,
            head_dim,
            head_dim,
            data['page_block_size'],
            layer_idx=torch.tensor(0, device=device), # a 0-D tensors
            causal=causal,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        
        output_per_head, lse_per_head = wrapper.run(query_per_head, (key_per_head, value_per_head))
        torch.cuda.synchronize()
        
        golden_outputs.append(output_per_head)
        golden_lses.append(lse_per_head)
    
    # Concatenate results
    golden_output = torch.cat(golden_outputs, dim=1)
    golden_lse = torch.cat(golden_lses, dim=1)
    
    # Assert with appropriate tolerances
    torch.testing.assert_close(output_wrapper, golden_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse_wrapper, golden_lse, rtol=1e-3, atol=1e-3)


# Additional test with custom sequence length ranges
@pytest.mark.parametrize("sparsity_ratio", [0.25, 0.5, 0.75])
def test_golden_reference_with_sparsity(sparsity_ratio, device="cuda"):
    """Test golden reference with different sparsity levels"""
    batch_size = 4
    num_kv_heads = 4
    num_qo_heads = 16
    head_dim = 128
    causal = True
    dtype = torch.bfloat16
    seed = 42
    
    data = generate_test_data(
        batch_size, num_kv_heads, num_qo_heads, head_dim,
        device, dtype, seed, sparsity_ratio=sparsity_ratio,
        min_seq_len=200, max_seq_len=1000,
        min_q_len=10, max_q_len=100
    )
    
    # Run wrapper
    batch_attention_wrapper = flashinfer.BatchAttentionWithPerHeadSelectPagedKVCacheWrapper(device=device)
    batch_attention_wrapper.plan(
        qo_indptr=data['qo_indptr'],
        kv_indptr=data['kv_indptr'],
        all_layers_per_head_kv_indices=data['all_layers_per_head_kv_indices'].int(), # (num_layers, num_kv_heads, total_kv_indices)
        seq_lens=data['sparsified_seq_lens'],
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        layer_idx=torch.tensor(0, device=device), # a 0-D tensors
        page_block_size=data['page_block_size'],
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
        use_triton=False
    )
    
    output_wrapper, lse_wrapper = batch_attention_wrapper.run(
        data['query'], (data['key'], data['value'])
    )
    
    # Compute golden reference
    gqa_group_size = num_qo_heads // num_kv_heads
    golden_outputs = []
    golden_lses = []
    
    for kv_head_idx in range(num_kv_heads):
        query_start = kv_head_idx * gqa_group_size
        query_end = (kv_head_idx + 1) * gqa_group_size
        query_per_head = data['query'][:, query_start:query_end, :].contiguous()
        
        head_kv_indices = data['all_layers_per_head_kv_indices'][0, kv_head_idx] # assuming that num_layers = 1
        key_per_head = data['key'][:, :, kv_head_idx:kv_head_idx+1, :].contiguous()
        value_per_head = data['value'][:, :, kv_head_idx:kv_head_idx+1, :].contiguous()
        
        wrapper = flashinfer.BatchAttention(kv_layout="NHD")
        wrapper.plan(
            data['qo_indptr'],
            data['kv_indptr'],
            head_kv_indices.int(),
            data['sparsified_seq_lens'],
            gqa_group_size,
            1,
            head_dim,
            head_dim,
            data['page_block_size'],
            layer_idx=torch.tensor(0, device=device), # a 0-D tensors
            causal=causal,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        
        output_per_head, lse_per_head = wrapper.run(query_per_head, (key_per_head, value_per_head))
        golden_outputs.append(output_per_head)
        golden_lses.append(lse_per_head)
    
    golden_output = torch.cat(golden_outputs, dim=1)
    golden_lse = torch.cat(golden_lses, dim=1)
    
    torch.testing.assert_close(output_wrapper, golden_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse_wrapper, golden_lse, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [4, 8, 16])
@pytest.mark.parametrize("num_kv_heads,num_qo_heads", [
    (4, 32),  # 1:8 GQA ratio
    (8, 32),  # 1:4 GQA ratio
])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_layers", [2, 4, 8, 16, 32])
def test_per_layer_correctness(
    batch_size, num_kv_heads, num_qo_heads, head_dim, 
    causal, dtype, num_layers, device="cuda"
):
    """Test whether the layer_idx is correctly updated"""
    seed = 42

    min_seq_len = 100
    max_seq_len = 500
    min_q_len = 1
    max_q_len = 50
    data = generate_test_data(
        batch_size, num_kv_heads, num_qo_heads, head_dim,
        device, dtype, seed,
        min_seq_len=min_seq_len, max_seq_len=max_seq_len,
        min_q_len=min_q_len, max_q_len=max_q_len, num_layers=num_layers
    )

    # golden, plan every time
    output_golden, lse_golden = [], []
    for layer_idx in range(num_layers):
        batch_attention_wrapper = flashinfer.BatchAttentionWithPerHeadSelectPagedKVCacheWrapper(device=device)
        per_head_kv_indices = data['all_layers_per_head_kv_indices'][layer_idx]
        batch_attention_wrapper.plan(
            qo_indptr=data['qo_indptr'],
            kv_indptr=data['kv_indptr'],
            all_layers_per_head_kv_indices=per_head_kv_indices.int(), # (num_layers, num_kv_heads, total_kv_indices)
            seq_lens=data['sparsified_seq_lens'],
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            layer_idx=torch.tensor(0, device=device), # a 0-D tensors
            page_block_size=data['page_block_size'],
            causal=causal,
            q_data_type=dtype,
            kv_data_type=dtype,
            use_triton=False,
            add_layer_idx_by_one_after_run=False
        )
        output_per_head, lse_per_head = batch_attention_wrapper.run(
            data['query'], (data['key'], data['value'])
        )
        output_golden.append(output_per_head)
        lse_golden.append(lse_per_head)

    
    # reference, plan once
    output_ours, lse_ours = [], []
    batch_attention_wrapper = flashinfer.BatchAttentionWithPerHeadSelectPagedKVCacheWrapper(device=device)
    layer_idx_buffer = torch.tensor([0], device=device)
    batch_attention_wrapper.plan(
        qo_indptr=data['qo_indptr'],
        kv_indptr=data['kv_indptr'],
        all_layers_per_head_kv_indices=data['all_layers_per_head_kv_indices'].int(), # (num_layers, num_kv_heads, total_kv_indices)
        seq_lens=data['sparsified_seq_lens'],
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        layer_idx=layer_idx_buffer, # a 0-D tensors
        page_block_size=data['page_block_size'],
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
        use_triton=False,
        add_layer_idx_by_one_after_run=True
    )
    for layer_idx in range(num_layers):
        output_per_head, lse_per_head = batch_attention_wrapper.run(
            data['query'], (data['key'], data['value'])
        )
        output_ours.append(output_per_head)
        lse_ours.append(lse_per_head)

    # check correctness
    for layer_idx in range(num_layers):
        torch.testing.assert_close(output_golden[layer_idx], output_ours[layer_idx], rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(lse_golden[layer_idx], lse_ours[layer_idx], rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize("num_layers", [2, 8, 16, 32])
def test_per_layer_correctness_with_cuda_graph(num_layers, device="cuda"):
    """Test whether the layer_idx is correctly updated with CUDA graph"""
    # -------------------------  test hyper-params  ------------------------- #
    seed = 42
    batch_size = 4
    num_kv_heads = 4
    num_qo_heads = 32
    head_dim = 128
    causal = True
    dtype = torch.bfloat16
    min_seq_len = 100
    max_seq_len = 500
    min_q_len = 1
    max_q_len = 50

    data = generate_test_data(
        batch_size, num_kv_heads, num_qo_heads, head_dim,
        device, dtype, seed,
        min_seq_len=min_seq_len, max_seq_len=max_seq_len,
        min_q_len=min_q_len, max_q_len=max_q_len, num_layers=num_layers
    )

    # golden, plan once, no cuda graph
    output_no_cuda_graph, lse_no_cuda_graph = [], []
    batch_attention_wrapper = flashinfer.BatchAttentionWithPerHeadSelectPagedKVCacheWrapper(device=device)
    layer_idx_buffer = torch.tensor([0], device=device)
    batch_attention_wrapper.plan(
        qo_indptr=data['qo_indptr'],
        kv_indptr=data['kv_indptr'],
        all_layers_per_head_kv_indices=data['all_layers_per_head_kv_indices'].int(), # (num_layers, num_kv_heads, total_kv_indices)
        seq_lens=data['sparsified_seq_lens'],
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        layer_idx=layer_idx_buffer, # a 0-D tensors
        page_block_size=data['page_block_size'],
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
        use_triton=False,
        add_layer_idx_by_one_after_run=True
    )
    for layer_idx in range(num_layers):
        output_per_head, lse_per_head = batch_attention_wrapper.run(
            data['query'], (data['key'], data['value'])
        )
        output_no_cuda_graph.append(output_per_head)
        lse_no_cuda_graph.append(lse_per_head)

    torch.cuda.synchronize()

    #===============================================

    # reference, plan once, with cuda graph
    batch_attention_wrapper_cg = flashinfer.BatchAttentionWithPerHeadSelectPagedKVCacheWrapper(device=device)
    layer_idx_buffer_cg = torch.tensor([0], device=device)
    batch_attention_wrapper_cg.plan(
        qo_indptr=data['qo_indptr'],
        kv_indptr=data['kv_indptr'],
        all_layers_per_head_kv_indices=data['all_layers_per_head_kv_indices'].int(), # (num_layers, num_kv_heads, total_kv_indices)
        seq_lens=data['sparsified_seq_lens'],
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        layer_idx=layer_idx_buffer_cg, # a 0-D tensors
        page_block_size=data['page_block_size'],
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
        use_triton=False,
        add_layer_idx_by_one_after_run=True
    )

    # warm-up in a side stream
    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream):
        for _ in range(3):
            layer_idx_buffer_cg.zero_()
            for _ in range(num_layers):
                batch_attention_wrapper_cg.run(data["query"], (data["key"], data["value"]))
    torch.cuda.current_stream().wait_stream(warm_stream)

    # allocate placeholder tensors that will *receive* results during capture
    cap_outputs = [torch.empty_like(output_no_cuda_graph[0]) for _ in range(num_layers)]
    cap_lses    = [torch.empty_like(lse_no_cuda_graph[0])    for _ in range(num_layers)]

    capture_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture_graph):
        layer_idx_buffer_cg.zero_()
        for i in range(num_layers):
            o, l = batch_attention_wrapper_cg.run(data["query"], (data["key"], data["value"]))
            cap_outputs[i].copy_(o)
            cap_lses[i].copy_(l)

    capture_graph.replay()
    torch.cuda.synchronize()

    # ------------------------- assertions ------------------------- #
    for i in range(num_layers):
        assert torch.allclose(
            output_no_cuda_graph[i], cap_outputs[i], rtol=2e-2, atol=2e-2
        ), f"output mismatch at layer {i}"
        assert torch.allclose(
            lse_no_cuda_graph[i], cap_lses[i], rtol=1e-3, atol=1e-3
        ), f"LSE mismatch at layer {i}"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
