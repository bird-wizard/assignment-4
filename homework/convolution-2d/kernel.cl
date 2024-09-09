__kernel void convolution2D(
    __global float * inputData, 
    __global float * outputData, 
    __constant float * maskData,
    int width, 
    int height, 
    int maskWidth,  
    int imageChannels){
    //@@ Insert code to implement matrix multiplication here

    // maskRadius := maskWidth/2 # this is integer division, so the result is 2
    const int maskRadius = maskWidth / 2;

    // for i from 0 to height do
    int row = get_global_id(1);
    
    // for j from 0 to width do
    int column = get_global_id(0);

    // for k from 0 to channels
    for (int k = 0; k < imageChannels; ++k) {

        // accum := 0
        float accum = 0.0f;

        // for y from -maskRadius to maskRadius do
        for (int y = -maskRadius; y <= maskRadius; ++y) {
            // for x from -maskRadius to maskRadius do
            for (int x = -maskRadius; x <= maskRadius; ++x) {
                // xOffset := j + x
                int xOffset = j + x;
                // yOffset := i + y
                int yOffset = i + y;

                // if xOffset >= 0 && xOffset < width && yOffset >= 0 && yOffset < height then
                if (xOffset >= 0 && xOffset < width && yOffset >= 0 && yOffset < height) {

                    // imagePixel := I[(yOffset * width + xOffset) * channels + k]
                    float imagePixel = inputData[(yOffset * width + xOffset) * imageChannels + k];

                    // maskValue := K[(y+maskRadius)*maskWidth+x+maskRadius]
                    float maskValue = maskData[(y + maskRadius) * maskWidth + (x + maskRadius)];

                    // accum += imagePixel * maskValue
                    accum += imagePixel * maskValue;
                }
            }
        }

        // min(max(x, lower), upper)
        accum = fmin(fmax(accum, 0.0f), 1.0f);
        // P[(i * width + j)*channels + k] = clamp(accum, 0, 1)
        outputData[(i * width + j) * imageChannels + k] = accum;
    }
}