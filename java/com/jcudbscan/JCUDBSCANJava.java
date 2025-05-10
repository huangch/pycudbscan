package com.jcudbscan;

/**
 * Java wrapper for CUDA DBSCAN implementation.
 */
public class JCUDBSCANJava {
    
    // Load the native library
    static {
        try {
            System.loadLibrary("jcudbscan_java");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Native code library failed to load: " + e);
            System.exit(1);
        }
    }
    
    /**
     * Check if CUDA is available.
     * 
     * @return true if CUDA is available, false otherwise
     */
    public native boolean checkCudaAvailable();
    
    /**
     * Execute a dummy CUDA function that multiplies each element by 2.
     * 
     * @param input the input array
     * @return the output array
     */
    public native float[] cudaDummyFunction(float[] input);
    
    /**
     * Simple example usage of the CUDA functions.
     */
    public static void main(String[] args) {
        JCUDBSCANJava cuda = new JCUDBSCANJava();
        
        // Check if CUDA is available
        boolean cudaAvailable = cuda.checkCudaAvailable();
        System.out.println("CUDA available: " + cudaAvailable);
        
        if (cudaAvailable) {
            // Create a sample array
            float[] input = new float[10];
            for (int i = 0; i < input.length; i++) {
                input[i] = i * 1.0f;
            }
            
            // Call the dummy function
            float[] output = cuda.cudaDummyFunction(input);
            
            // Print the results
            System.out.println("Results:");
            for (int i = 0; i < input.length; i++) {
                System.out.printf("Input: %.1f, Output: %.1f\n", input[i], output[i]);
            }
        }
    }
}