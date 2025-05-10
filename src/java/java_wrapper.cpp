#include <jni.h>
#include <cuda_runtime.h>

// Forward declarations of CUDA functions (defined in main.cu)
extern "C" {
    bool check_cuda_available();
    void cuda_dummy_function(float* input, float* output, int size);
}

// JNI wrapper for checking CUDA availability
extern "C" JNIEXPORT jboolean JNICALL
Java_com_jcudbscan_JCUDBSCANJava_checkCudaAvailable(JNIEnv* env, jobject obj) {
    return check_cuda_available();
}

// JNI wrapper for the dummy CUDA function
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_jcudbscan_JCUDBSCANJava_cudaDummyFunction(JNIEnv* env, jobject obj, jfloatArray input) {
    // Get the input array
    jsize length = env->GetArrayLength(input);
    jfloat* inputData = env->GetFloatArrayElements(input, NULL);
    
    // Create output array
    jfloatArray output = env->NewFloatArray(length);
    jfloat* outputData = env->GetFloatArrayElements(output, NULL);
    
    // Call the CUDA function
    cuda_dummy_function((float*)inputData, (float*)outputData, length);
    
    // Release the arrays
    env->ReleaseFloatArrayElements(input, inputData, JNI_ABORT);
    env->ReleaseFloatArrayElements(output, outputData, 0);
    
    return output;
}