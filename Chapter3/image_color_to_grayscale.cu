#include <opencv2/opencv.hpp>

#define NUM_CHANNELS 3

__global__
void color_to_grayscale_conversion(unsigned char* in, unsigned char* out, int width, int height){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < 0 || row >= height || col < 0 || col >= width) return;

    int grey_offset = row * width + col;
    int rgb_offset = grey_offset * NUM_CHANNELS;

    unsigned char r = in[rgb_offset + 0];
    unsigned char g = in[rgb_offset + 1];
    unsigned char b = in[rgb_offset + 2];

    out[grey_offset] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
}

int main(int argc, char* argv[]){

    int block_dim = 16; // Ajusta según tus necesidades
    int image_width;
    int image_height;
    int size;
    unsigned char* h_input_image, *h_output_image;
    unsigned char* d_input_image, *d_output_image;

    if(argc != 2){
        printf("Usage: %s <image_path>\n", argv[0]);
        return 1;
    }
    
    // Cargar la imagen desde un archivo usando OpenCV
    cv::Mat input_image = cv::imread(argv[1]);

    if (!input_image.data) {
        std::cerr << "Error al cargar la imagen desde " << argv[1] << std::endl;
        return 1;
    }

    image_width = input_image.cols;
    image_height = input_image.rows;
    size = image_width * image_height;

    // Allocate memory for the input and output images on host
    h_input_image = (unsigned char*) malloc(NUM_CHANNELS * size * sizeof(unsigned char));
    h_output_image = (unsigned char*) malloc(size * sizeof(unsigned char));

    // Copy pixel values from the OpenCV image to the input array
    memcpy(h_input_image, input_image.data, NUM_CHANNELS * size * sizeof(unsigned char));

    // Allocate memory for the input and output images on device
    cudaMalloc((void**) &d_input_image, NUM_CHANNELS * size * sizeof(unsigned char));
    cudaMalloc((void**) &d_output_image, size * sizeof(unsigned char));

    // Copy the input image to the device
    cudaMemcpy(d_input_image, h_input_image, NUM_CHANNELS * size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Convertir la imagen a escala de grises
    dim3 dimBlock(block_dim, block_dim, 1);
    dim3 dimGrid(ceil((float)image_width/block_dim), ceil((float)image_height/block_dim), 1);
    color_to_grayscale_conversion<<<dimGrid, dimBlock>>>(d_input_image, d_output_image, image_width, image_height);

    // Copiar la salida de nuevo al host
    cudaMemcpy(h_output_image, d_output_image, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Guardar la imagen resultante usando OpenCV
    cv::Mat output_image(image_height, image_width, CV_8UC1, h_output_image);
    cv::imwrite("output_image.jpg", output_image);

    // Mostrar la imagen resultante usando OpenCV
    cv::imshow("Output Image", output_image);
    cv::waitKey(0); // Esperar hasta que se presione una tecla

    // Liberar la memoria del dispositivo
    cudaFree(d_input_image);
    cudaFree(d_output_image);

    // Liberar la memoria del host
    free(h_input_image);
    free(h_output_image);

    return 0;
}