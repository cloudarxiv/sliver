#ifndef _KERNEL_WRAPPER_H
#define _KERNEL_WRAPPER_H

#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>

#include "Scheduler.h"
#include "KernelControlBlock.h"
#include "Kernel.h"
#include "errchk.h"

class KernelWrapper
{
public:
    KernelWrapper(Scheduler *scheduler,
                  CUcontext &context,
                  const std::string &moduleFile,
                  const std::string &kernelName,
                  kernel_attr_t *attr,
                  Kernel *kernel) : scheduler(scheduler),
                                    context(context),
                                    moduleFile(moduleFile),
                                    kernelName(kernelName),
                                    attr(attr),
                                    kernel(kernel)
    {
        attr->kernelParams = kernel->args;

        kernel->args[0] = &(attr->blockOffsetX);
        kernel->args[1] = &(attr->blockOffsetY);
        kernel->args[2] = &(attr->blockOffsetZ);

        kernel_control_block_init(&(attr->kcb), (attr->gridDimX * attr->gridDimY * attr->gridDimZ) / (attr->sGridDimX * attr->sGridDimY * attr->sGridDimZ));
    }

    ~KernelWrapper() {}

    void setNiceValue(unsigned int niceness = 0)
    {
        attr->niceness = niceness;
    }

    void launch()
    {
        pthread_create(&thread, NULL, threadFunction, this);
    }

    void finish()
    {
        pthread_join(thread, NULL);
        kernel_control_block_destroy(&(attr->kcb));
    }

    kernel_attr_t *getKernelAttributes() { return attr; }

private:
    CUcontext context;
    pthread_t thread;
    std::string moduleFile;
    std::string kernelName;
    CUmodule module;

    kernel_attr_t *attr;

    Kernel *kernel;
    Scheduler *scheduler;

    void *threadFunction()
    {
        struct timeval t0, t1, dt;

        gettimeofday(&t0, NULL);

        checkCudaErrors(cuCtxSetCurrent(context));
        checkCudaErrors(cuModuleLoad(&module, moduleFile.c_str()));
        checkCudaErrors(cuModuleGetFunction(&(attr->function), module, kernelName.c_str()));

        kernel->memAlloc();
        kernel->memcpyHtoD(attr->stream);

        set_state(&(attr->kcb), MEMCPYHTOD);

        scheduler->scheduleKernel(this->attr);

        pthread_mutex_lock(&(attr->kcb.kernel_lock));
        while (attr->kcb.state != MEMCPYDTOH)
        {
            pthread_cond_wait(&(attr->kcb.kernel_signal), &(attr->kcb.kernel_lock));
        }
        pthread_mutex_unlock(&(attr->kcb.kernel_lock));

        kernel->memcpyDtoH(attr->stream);
        kernel->memFree();

        gettimeofday(&t1, NULL);
        timersub(&t1, &t0, &dt);
        printf("[thread id: %ld\tkernel id: %d\tniceness: %d] done in %ld.%06ldsec\n", pthread_self(), attr->id, attr->niceness, dt.tv_sec, dt.tv_usec);
        return nullptr;
    }

    static void *threadFunction(void *args)
    {
        KernelWrapper *kernelWrapper = static_cast<KernelWrapper *>(args);
        return kernelWrapper->threadFunction();
    }
};

#endif
