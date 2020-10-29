def correlation_forward_cuda_kernel(output,
                                    ob,
                                    oc,
                                    oh,
                                    ow,
                                    osb,
                                    osc,
                                    osh,
                                    osw,

                                    input1,
                                    ic,
                                    ih,
                                    iw,
                                    isb,
                                    isc,
                                    ish,
                                    isw,

                                    input2,
                                    gc,
                                    gsb,
                                    gsc,
                                    gsh,
                                    gsw,

                                    rInput1,
                                    rInput2,
                                    pad_size,
                                    kernel_size,
                                    max_displacement,
                                    stride1,
                                    stride2,
                                    corr_type_multiply,

                                    ):
    pass


def correlation_backward_cuda_kernel(
        gradOutput,
        gob,
        goc,
        goh,
        gow,
        gosb,
        gosc,
        gosh,
        gosw,

        input1,
        ic,
        ih,
        iw,
        isb,
        isc,
        ish,
        isw,

        input2,
        gsb,
        gsc,
        gsh,
        gsw,

        gradInput1,
        gisb,
        gisc,
        gish,
        gisw,

        gradInput2,
        ggc,
        ggsb,
        ggsc,
        ggsh,
        ggsw,

        rInput1,
        rInput2,
        pad_size,
        kernel_size,
        max_displacement,
        stride1,
        stride2,
        corr_type_multiply,

):
    pass
