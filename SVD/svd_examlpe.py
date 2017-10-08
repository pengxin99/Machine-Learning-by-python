def tsqr(data, name=None, compute_svd=False):
    """ Direct Tall-and-Skinny QR algorithm

    As presented in:

        A. Benson, D. Gleich, and J. Demmel.
        Direct QR factorizations for tall-and-skinny matrices in
        MapReduce architectures.
        IEEE International Conference on Big Data, 2013.
        http://arxiv.org/abs/1301.1071

    This algorithm is used to compute both the QR decomposition and the
    Singular Value Decomposition.  It requires that the input array have a
    single column of blocks, each of which fit in memory.

    If blocks are of size ``(n, k)`` then this algorithm has memory use that
    scales as ``n**2 * k * nthreads``.

    Parameters
    ----------

    data: Array
    compute_svd: bool
        Whether to compute the SVD rather than the QR decomposition

    See Also
    --------

    dask.array.linalg.qr - Powered by this algorithm
    dask.array.linalg.svd - Powered by this algorithm
    """

    if not (data.ndim == 2 and                    # Is a matrix
            len(data.chunks[1]) == 1):         # Only one column block
        raise ValueError(
            "Input must have the following properties:\n"
            "  1. Have two dimensions\n"
            "  2. Have only one column of blocks")

    prefix = name or 'tsqr-' + tokenize(data, compute_svd)
    prefix += '_'

    m, n = data.shape
    numblocks = (len(data.chunks[0]), 1)

    name_qr_st1 = prefix + 'QR_st1'
    dsk_qr_st1 = top(np.linalg.qr, name_qr_st1, 'ij', data.name, 'ij',
                     numblocks={data.name: numblocks})
    # qr[0]
    name_q_st1 = prefix + 'Q_st1'
    dsk_q_st1 = dict(((name_q_st1, i, 0),
                      (operator.getitem, (name_qr_st1, i, 0), 0))
                     for i in range(numblocks[0]))
    # qr[1]
    name_r_st1 = prefix + 'R_st1'
    dsk_r_st1 = dict(((name_r_st1, i, 0),
                      (operator.getitem, (name_qr_st1, i, 0), 1))
                     for i in range(numblocks[0]))

    # Stacking for in-core QR computation
    to_stack = [(name_r_st1, i, 0) for i in range(numblocks[0])]
    name_r_st1_stacked = prefix + 'R_st1_stacked'
    dsk_r_st1_stacked = {(name_r_st1_stacked, 0, 0): (np.vstack,
                                                      (tuple, to_stack))}
    # In-core QR computation
    name_qr_st2 = prefix + 'QR_st2'
    dsk_qr_st2 = top(np.linalg.qr, name_qr_st2, 'ij', name_r_st1_stacked, 'ij',
                     numblocks={name_r_st1_stacked: (1, 1)})
    # qr[0]
    name_q_st2_aux = prefix + 'Q_st2_aux'
    dsk_q_st2_aux = {(name_q_st2_aux, 0, 0): (operator.getitem,
                                              (name_qr_st2, 0, 0), 0)}
    if not any(np.isnan(c) for cs in data.chunks for c in cs):
        q2_block_sizes = [min(e, n) for e in data.chunks[0]]
        block_slices = [(slice(e[0], e[1]), slice(0, n))
                        for e in _cumsum_blocks(q2_block_sizes)]
        dsk_q_blockslices = {}
    else:
        name_q2bs = prefix + 'q2-shape'
        dsk_q2_shapes = {(name_q2bs, i): (min, (getattr, (data.name, i, 0), 'shape'))
                         for i in range(numblocks[0])}
        dsk_n = {prefix + 'n': (operator.getitem,
                                (getattr, (data.name, 0, 0), 'shape'), 1)}
        name_q2cs = prefix + 'q2-shape-cumsum'
        dsk_q2_cumsum = {(name_q2cs, 0): [0, (name_q2bs, 0)]}
        dsk_q2_cumsum.update({(name_q2cs, i): (_cumsum_part,
                                               (name_q2cs, i - 1),
                                               (name_q2bs, i))
                              for i in range(1, numblocks[0])})

        name_blockslice = prefix + 'q2-blockslice'
        dsk_block_slices = {(name_blockslice, i): (tuple, [
            (apply, slice, (name_q2cs, i)), (slice, 0, prefix + 'n')])
            for i in range(numblocks[0])}

        dsk_q_blockslices = merge(dsk_n,
                                  dsk_q2_shapes,
                                  dsk_q2_cumsum,
                                  dsk_block_slices)

        block_slices = [(name_blockslice, i) for i in range(numblocks[0])]

    name_q_st2 = prefix + 'Q_st2'
    dsk_q_st2 = dict(((name_q_st2, i, 0),
                      (operator.getitem, (name_q_st2_aux, 0, 0), b))
                     for i, b in enumerate(block_slices))
    # qr[1]
    name_r_st2 = prefix + 'R'
    dsk_r_st2 = {(name_r_st2, 0, 0): (operator.getitem, (name_qr_st2, 0, 0), 1)}

    name_q_st3 = prefix + 'Q'
    dsk_q_st3 = top(np.dot, name_q_st3, 'ij', name_q_st1, 'ij',
                    name_q_st2, 'ij', numblocks={name_q_st1: numblocks,
                                                 name_q_st2: numblocks})

    dsk_q = {}
    dsk_q.update(data.dask)
    dsk_q.update(dsk_qr_st1)
    dsk_q.update(dsk_q_st1)
    dsk_q.update(dsk_r_st1)
    dsk_q.update(dsk_r_st1_stacked)
    dsk_q.update(dsk_qr_st2)
    dsk_q.update(dsk_q_st2_aux)
    dsk_q.update(dsk_q_st2)
    dsk_q.update(dsk_q_st3)
    dsk_q.update(dsk_q_blockslices)
    dsk_r = {}
    dsk_r.update(data.dask)
    dsk_r.update(dsk_qr_st1)
    dsk_r.update(dsk_r_st1)
    dsk_r.update(dsk_r_st1_stacked)
    dsk_r.update(dsk_qr_st2)
    dsk_r.update(dsk_r_st2)

    if not compute_svd:
        qq, rr = np.linalg.qr(np.ones(shape=(1, 1), dtype=data.dtype))
        q = Array(dsk_q, name_q_st3, shape=data.shape, chunks=data.chunks,
                  dtype=qq.dtype)
        r = Array(dsk_r, name_r_st2, shape=(n, n), chunks=(n, n),
                  dtype=rr.dtype)
        return q, r
    else:
        # In-core SVD computation
        name_svd_st2 = prefix + 'SVD_st2'
        dsk_svd_st2 = top(np.linalg.svd, name_svd_st2, 'ij', name_r_st2, 'ij',
                          numblocks={name_r_st2: (1, 1)})
        # svd[0]
        name_u_st2 = prefix + 'U_st2'
        dsk_u_st2 = {(name_u_st2, 0, 0): (operator.getitem,
                                          (name_svd_st2, 0, 0), 0)}
        # svd[1]
        name_s_st2 = prefix + 'S'
        dsk_s_st2 = {(name_s_st2, 0): (operator.getitem,
                                       (name_svd_st2, 0, 0), 1)}
        # svd[2]
        name_v_st2 = prefix + 'V'
        dsk_v_st2 = {(name_v_st2, 0, 0): (operator.getitem,
                                          (name_svd_st2, 0, 0), 2)}
        # Q * U
        name_u_st4 = prefix + 'U'
        dsk_u_st4 = top(dotmany, name_u_st4, 'ij', name_q_st3, 'ik',
                        name_u_st2, 'kj', numblocks={name_q_st3: numblocks,
                                                     name_u_st2: (1, 1)})

        dsk_u = {}
        dsk_u.update(dsk_q)
        dsk_u.update(dsk_r)
        dsk_u.update(dsk_svd_st2)
        dsk_u.update(dsk_u_st2)
        dsk_u.update(dsk_u_st4)
        dsk_s = {}
        dsk_s.update(dsk_r)
        dsk_s.update(dsk_svd_st2)
        dsk_s.update(dsk_s_st2)
        dsk_v = {}
        dsk_v.update(dsk_r)
        dsk_v.update(dsk_svd_st2)
        dsk_v.update(dsk_v_st2)

        uu, ss, vv = np.linalg.svd(np.ones(shape=(1, 1), dtype=data.dtype))

        u = Array(dsk_u, name_u_st4, shape=data.shape, chunks=data.chunks,
                  dtype=uu.dtype)
        s = Array(dsk_s, name_s_st2, shape=(n,), chunks=((n,),), dtype=ss.dtype)
        v = Array(dsk_v, name_v_st2, shape=(n, n), chunks=((n,), (n,)),
                  dtype=vv.dtype)
        return u, s, v