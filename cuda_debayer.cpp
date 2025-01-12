#include <stdio.h>
#include <string>
#include <iostream>
#include <memory>
#include <valarray>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <CCfits/CCfits>

#include "kernel.h"

#define PROJECT_NAME "cuda_debayer"

struct Image {
    std::vector<uint8_t> img;
    int width;
    int height;
};

std::vector<uint8_t> to_8bit(auto input) {
    std::vector<uint8_t> output;
    output.reserve(input.size());
    for (auto it = std::begin(input); it != std::end(input); ++it) {
        uint8_t pix = (*it)>>8;
        output.push_back(pix);
    }
    return output;
}

Image read_image(std::string img_fname) {
    auto pInfile = std::make_unique<CCfits::FITS>(img_fname,CCfits::Read,true);
    // Access the primary HDU
    CCfits::PHDU& imageHDU = pInfile->pHDU();

    // Get image dimensions
    long naxis1 = imageHDU.axis(0); 
    long naxis2 = imageHDU.axis(1);

    // Create a valarray to store the image data
    std::valarray<uint16_t> imageData(naxis1 * naxis2);

    // Read the image data
    imageHDU.read(imageData);

    auto result = to_8bit(imageData);
    return {result, (int)naxis1, (int)naxis2};
}



int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    
    initializeCUDA();

    std::string fname = "image.fits";
    std::string output_fname = "output";
    auto image = read_image(fname);
    printf("Read image with size: %d, %d\n",image.width, image.height); 
    
    uint8_t* debayered = (uint8_t*)malloc(image.img.size()*sizeof(image.img[0])*3);
    std::string bayer_fname = output_fname + "_RGGB.png";
    demosaicImage(image.img.data(), debayered, image.width, image.height, BayerPattern::RGGB);
    stbi_write_png(bayer_fname.c_str(), image.width, image.height, 3, debayered, image.width*3);

    bayer_fname = output_fname + "_BGGR.png";
    demosaicImage(image.img.data(), debayered, image.width, image.height, BayerPattern::BGGR);
    stbi_write_png(bayer_fname.c_str(), image.width, image.height, 3, debayered, image.width*3);

    bayer_fname = output_fname + "_GRBG.png";
    demosaicImage(image.img.data(), debayered, image.width, image.height, BayerPattern::GRBG);
    stbi_write_png(bayer_fname.c_str(), image.width, image.height, 3, debayered, image.width*3);

    bayer_fname = output_fname + "_GBRG.png";
    demosaicImage(image.img.data(), debayered, image.width, image.height, BayerPattern::GBRG);
    stbi_write_png(bayer_fname.c_str(), image.width, image.height, 3, debayered, image.width*3);

    free(debayered);
    printf("This is project %s.\n", PROJECT_NAME);
    return 0;
}
