# Assignment 4

## Homework
### Convolution 2D
For this assignment, I created a 3-dimensional openCL buffer for the input image, a 2-dimensional buffer for the mask matrix, and another 3-dimensional openCL buffer for the output image. Because we were treating this as a 2D convolution, I set the global item size to a 2-dimensional array of the size of the input image. I launched the kernel as usual with clEnqueueNDRangeKernel and read the buffer back as a 3-dimensional matrix. The pseudocode did most of the heavy lifting with the kernel code so that made the execution pretty straight forward. As a path forward, I would expect to use local memory as a method of optimization