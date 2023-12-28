#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

#define DEBUG

#define BLUR_SIZE 5
#define BLOCK_SIZE 16

__global__
void blur_kernel(unsigned char* in, unsigned char* out, int width, int height, int channels){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(col >= width || row >= height) return;

    for (int channel = 0; channel < channels; ++channel) {
        int pixVal = 0;
        int pixels = 0;

        // Compute the average of the neighboring pixels
        for(int blurrow = -BLUR_SIZE; blurrow < BLUR_SIZE + 1; ++blurrow){
            for(int blurcol = -BLUR_SIZE; blurcol < BLUR_SIZE + 1; ++blurcol){
                int currow = row + blurrow;
                int curcol = col + blurcol;

                // Check if the current pixel is in the image
                if(curcol < 0 || curcol >= width || currow < 0 || currow >= height) continue;

                pixVal += in[(currow * width + curcol) * channels + channel];
                ++pixels; // Count the number of pixel values that have been added
            }
        }

        // Write out the result for this pixel and channel
        out[(row * width + col) * channels + channel] = (unsigned char) ((float)pixVal / pixels);
    }
}

int main(int argc, char* argv[]){

    int image_width;
    int image_height;
    int channels;
    int size;
    unsigned char* h_input_image, *h_output_image;
    unsigned char* d_input_image, *d_output_image;

    if(argc != 3){
        printf("Usage: %s <input_image_path> <output_image_path>\n", argv[0]);
        return 1;
    }

    char* input_image_path = argv[1];
    char* output_image_path = argv[2];

    // Load the input image using OpenCV as a color image
    cv::Mat image = cv::imread(input_image_path, cv::IMREAD_COLOR);

    if (image.empty()) {
        fprintf(stderr, "Error: Unable to load the image.\n");
        return 1;
    }

    image_width = image.cols;
    image_height = image.rows;
    channels = image.channels();
    size = image_width * image_height * channels;

    // Allocate memory for the input and output images on host
    h_input_image = (unsigned char*) malloc(size * sizeof(unsigned char));
    h_output_image = (unsigned char*) malloc(size * sizeof(unsigned char));

    // Allocate memory for the input and output images on device
    cudaMalloc((void**) &d_input_image, size * sizeof(unsigned char));
    cudaMalloc((void**) &d_output_image, size * sizeof(unsigned char));

    // Copy the input image data to the host buffer
    memcpy(h_input_image, image.data, size * sizeof(unsigned char));

    // Copy the input image to the device
    cudaMemcpy(d_input_image, h_input_image, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Blur the image
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dimGrid(ceil((float)image_width/BLOCK_SIZE), ceil((float)image_height/BLOCK_SIZE), 1);
    blur_kernel<<<dimGrid, dimBlock>>>(d_input_image, d_output_image, image_width, image_height, channels);

    // Copy the output back to the host
    cudaMemcpy(h_output_image, d_output_image, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save the output image using OpenCV
    cv::Mat output_image(image_height, image_width, CV_8UC3, h_output_image);
    cv::imwrite(output_image_path, output_image);

    // Free the device memory
    cudaFree(d_input_image);
    cudaFree(d_output_image);

    // Free the host memory
    free(h_input_image);
    free(h_output_image);

    return 0;
}
