#pragma once

enum BayerPattern {
    RGGB = 0,
    BGGR,
    GRBG,
    GBRG

};
void initializeCUDA();

void demosaicImage(const unsigned char* bayerImage, unsigned char* outputImage, int width, int height, BayerPattern pattern);

