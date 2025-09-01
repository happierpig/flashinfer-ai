import pytest
import torch
import random
import numpy as np

import flashinfer


@pytest.mark.parametrize("contiguous", [True, False])
def test_append_paged_kv_cache(contiguous):
    nnz_kv = 100
    num_kv_heads = 32
    head_dim = 128

    if contiguous:
        k_append = torch.randn(nnz_kv, num_kv_heads, head_dim).half().to(0)
        v_append = torch.randn(nnz_kv, num_kv_heads, head_dim).half().to(0)
    else:
        kv_append = torch.randn(nnz_kv, 2, num_kv_heads, head_dim).half().to(0)
        k_append = kv_append[:, 0]
        v_append = kv_append[:, 1]
    # 45 + 8 + 25 + 22 = nnz_kv
    kv_append_length = torch.tensor([45, 8, 25, 22], dtype=torch.int32, device="cuda:0")
    kv_append_indptr = torch.cat(
        [torch.zeros(1).int().to(0), torch.cumsum(kv_append_length, dim=0)]
    ).int()

    max_num_pages = 1000
    page_size = 16
    paged_kv_cache = (
        torch.randn(max_num_pages, 2, page_size, num_kv_heads, head_dim).half().to(0)
    )
    num_pages_per_req = torch.tensor([3, 1, 2, 2], dtype=torch.int32, device="cuda:0")
    kv_page_indptr = torch.cat(
        [torch.zeros(1).int().to(0), torch.cumsum(num_pages_per_req, dim=0)]
    ).int()
    # use first 8 pages in the paged-kv
    kv_page_indices = torch.arange(8, dtype=torch.int32, device="cuda:0")
    # 45 = (3 - 1) * 16 + 13
    # 8 = (1 - 1) * 16 + 8
    # 25 = (2 - 1) * 16 + 9
    # 22 = (2 - 1) * 16 + 6
    kv_last_page_len = torch.tensor([13, 8, 9, 6], dtype=torch.int32, device="cuda:0")
    batch_indices, positions = flashinfer.get_batch_indices_positions(
        kv_append_indptr,
        flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, page_size),
        nnz_kv,
    )

    flashinfer.append_paged_kv_cache(
        k_append,
        v_append,
        batch_indices,
        positions,
        paged_kv_cache,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_len,
    )


@pytest.mark.parametrize("batch_size", [8, 15, 61])
@pytest.mark.parametrize("nnz_kv", [100, 500, 1000])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8, 16])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("use_graph", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_append_paged_kv_cache_graph(
    batch_size, nnz_kv, num_kv_heads, head_dim, use_graph, contiguous
):
    if contiguous:
        k_append = torch.randn(nnz_kv, num_kv_heads, head_dim).half().to(0)
        v_append = torch.randn(nnz_kv, num_kv_heads, head_dim).half().to(0)
    else:
        kv_append = torch.randn(nnz_kv, 2, num_kv_heads, head_dim).half().to(0)
        k_append = kv_append[:, 0]
        v_append = kv_append[:, 1]

    mid_points = random.sample(range(1, nnz_kv), batch_size - 1)
    mid_points.sort()
    kv_append_indptr = (
        torch.from_numpy(np.concatenate([[0], mid_points, [nnz_kv]])).int().to(0)
    )

    max_num_pages = 2048
    page_size = 16
    paged_kv_cache = (
        torch.randn(max_num_pages, 2, page_size, num_kv_heads, head_dim).half().to(0)
    )
    paged_kv_cache_ref = paged_kv_cache.clone()
    if max_num_pages * page_size <= nnz_kv:
        pytest.skip("max_num_pages * page_size <= nnz_kv")

    num_pages_per_req = (
        kv_append_indptr[1:] - kv_append_indptr[:-1] + page_size - 1
    ) // page_size
    kv_page_indptr = torch.cat(
        [torch.zeros(1).int().to(0), torch.cumsum(num_pages_per_req, dim=0)]
    ).int()
    kv_page_indices = torch.arange(
        kv_page_indptr[-1].item(), dtype=torch.int32, device="cuda:0"
    )
    kv_last_page_len = (
        kv_append_indptr[1:] - kv_append_indptr[:-1] - 1
    ) % page_size + 1
    batch_indices, positions = flashinfer.get_batch_indices_positions(
        kv_append_indptr,
        flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, page_size),
        nnz_kv,
    )

    if not use_graph:
        flashinfer.append_paged_kv_cache_graph(
            k_append,
            v_append,
            batch_indices,
            positions,
            torch.tensor([nnz_kv], dtype=torch.int32, device="cuda:0"),
            paged_kv_cache,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_len,
        )
    else:
        max_token_nnz = 1024 * 10
        batch_indices_buf = torch.empty(
            max_token_nnz, dtype=torch.int32, device="cuda:0"
        )
        positions_buf = torch.empty(max_token_nnz, dtype=torch.int32, device="cuda:0")
        kv_page_indices_buf = torch.empty(
            max_token_nnz, dtype=torch.int32, device="cuda:0"
        )
        kv_page_indptr_buf = torch.empty(
            max_token_nnz, dtype=torch.int32, device="cuda:0"
        )
        kv_last_page_len_buf = torch.empty(
            max_token_nnz, dtype=torch.int32, device="cuda:0"
        )
        nnz_buf = torch.tensor([0], dtype=torch.int32, device="cuda:0")

        # capture random input
        capture_nnz = nnz_kv // 2
        capture_batch_size = batch_size // 2

        mid_points = random.sample(range(1, capture_nnz), capture_batch_size - 1)
        mid_points.sort()
        capture_kv_append_indptr = (
            torch.from_numpy(np.concatenate([[0], mid_points, [capture_nnz]]))
            .int()
            .to(0)
        )

        num_pages_per_req = (
            capture_kv_append_indptr[1:] - capture_kv_append_indptr[:-1] + page_size - 1
        ) // page_size
        capture_kv_page_indptr = torch.cat(
            [torch.zeros(1).int().to(0), torch.cumsum(num_pages_per_req, dim=0)]
        ).int()
        capture_kv_page_indices = torch.arange(
            capture_kv_page_indptr[-1].item(), dtype=torch.int32, device="cuda:0"
        )
        capture_kv_last_page_len = (
            capture_kv_append_indptr[1:] - capture_kv_append_indptr[:-1] - 1
        ) % page_size + 1
        capture_batch_indices, capture_positions = (
            flashinfer.get_batch_indices_positions(
                capture_kv_append_indptr,
                flashinfer.get_seq_lens(
                    capture_kv_page_indptr, capture_kv_last_page_len, page_size
                ),
                capture_nnz,
            )
        )

        # copy to buf
        batch_indices_buf[: capture_batch_indices.size(0)] = capture_batch_indices
        positions_buf[: capture_positions.size(0)] = capture_positions
        kv_page_indices_buf[: capture_kv_page_indices.size(0)] = capture_kv_page_indices
        kv_page_indptr_buf[: capture_kv_page_indptr.size(0)] = capture_kv_page_indptr
        kv_last_page_len_buf[: capture_kv_last_page_len.size(0)] = (
            capture_kv_last_page_len
        )
        nnz_buf[0] = capture_nnz

        # capture graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            flashinfer.append_paged_kv_cache_graph(
                k_append,
                v_append,
                batch_indices_buf[: capture_batch_indices.size(0)],
                positions_buf[: capture_positions.size(0)],
                nnz_buf,
                paged_kv_cache,
                kv_page_indices_buf[: capture_kv_page_indices.size(0)],
                kv_page_indptr_buf[: capture_kv_page_indptr.size(0)],
                kv_last_page_len_buf[: capture_kv_last_page_len.size(0)],
            )

        # setup replay input
        paged_kv_cache.copy_(paged_kv_cache_ref)
        batch_indices_buf[: batch_indices.size(0)] = batch_indices
        positions_buf[: positions.size(0)] = positions
        kv_page_indices_buf[: kv_page_indices.size(0)] = kv_page_indices
        kv_page_indptr_buf[: kv_page_indptr.size(0)] = kv_page_indptr
        kv_last_page_len_buf[: kv_last_page_len.size(0)] = kv_last_page_len
        nnz_buf[0] = nnz_kv

        # replay
        g.replay()

    # calculate ref output
    flashinfer.append_paged_kv_cache(
        k_append,
        v_append,
        batch_indices,
        positions,
        paged_kv_cache_ref,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_len,
    )
    torch.testing.assert_close(paged_kv_cache, paged_kv_cache_ref)
