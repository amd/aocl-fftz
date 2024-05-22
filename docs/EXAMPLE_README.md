@page API_Usage API Usage

The following sample program shows the usage of the aocl-fftz APIs to setup and execute an fft problem.

---
@note Please ensure that you utilize the appropriate compatible data type/model APIs.\n ex:- For a problem descriptor of type FLOAT LP64 The following would be the API call sequence\n aoclfftz_setup_f\n aoclfftz_execute_f/ aoclfftz_destroy_f
---

```C
#include <stdio.h>
#include <stdlib.h>
#include "aoclfftz.h"

int main()
{
    // Create and initialize prob_desc params
    aoclfftz_prob_desc_d *problem =
        (aoclfftz_prob_desc_d *)malloc(sizeof(aoclfftz_prob_desc_d));
    problem->dim_rank = 1;
    problem->vec_rank = 1;
    problem->dims =
        (aoclfftz_dim_t *)malloc(sizeof(aoclfftz_dim_t) * problem->dim_rank);
    problem->vecs =
        (aoclfftz_dim_t *)malloc(sizeof(aoclfftz_dim_t) * problem->vec_rank);
    problem->flags = 0b0000; // complex, forward, in-order, in-place problem
    problem->dims[0].n = 10;
    problem->dims[0].in_stride = 1;
    problem->dims[0].out_stride = 1;
    problem->vecs[0].n = 1;
    problem->vecs[0].in_stride = 1;
    problem->vecs[0].out_stride = 1;
    problem->pthr_fft.dynamic_load_model = 0;
    problem->pthr_fft.num_threads = 0;
    problem->cntrl_params.logger_mode = 0;
    problem->cntrl_params.measure_stats = 0;
    problem->cntrl_params.opt_level = -1;
    problem->cntrl_params.opt_off = 1;

    DOUBLE in[20] = {1, 1, 2, 2, 3, 3, 4, 4,  5,  5,
                     6, 6, 7, 7, 8, 8, 9, 9, 10, 10};
    DOUBLE out[20] = {};
    problem->in = in;
    problem->out = out;

    // setup call
    VOID *aoclfftz_handle = aoclfftz_setup_d(problem);

    if (aoclfftz_handle == NULL)
    {
        printf("\nSetup Failure\n");
        goto exit_api;
    }
    printf("\nSetup succesful\n");

    // execute call
    INT32 res = aoclfftz_execute_d(aoclfftz_handle);

    if (res == AOCLFFTZ_EXECUTION_FAILURE)
    {
        printf("\nExecution Failure\n");
        goto exit_api;
    }
    printf("\nExecution succesful\n");

    exit_api:
    // destroy handle
    aoclfftz_destroy_d(aoclfftz_handle);
    free(problem->dims);
    free(problem->vecs);
    free(problem);

    return 0;
}
```

To build this example test program on a Linux system using GCC or AOCC, you must specify the aoclfftz.h header file and link the libaocl_fftz.so file as follows:

`gcc -I<aoclfftz.h file path> -L<libaocl_fftz.so file path> test.c -laocl_fftz -lm`
