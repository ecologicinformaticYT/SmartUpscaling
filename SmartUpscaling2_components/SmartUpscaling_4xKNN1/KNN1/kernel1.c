/*
Kernel OpenCL for SmartUpscaling_4xKNN1
Based on SmartUpcaling_KNN1
*/

#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))

__kernel void upscale(__global uchar4 *src, __global uchar4 *dst, float src_width, float src_height, float dst_width, float dst_height, float aa_factor){
    /*
    Kernel OpenCL for SmartUpscaling_KNN1 (improved version).
    It uses a trained version of the k nearest neighbors algorithm.
    Parameters:
    src: source image
    dst: destination image
    src_width: width of source image
    src_height: height of source image
    dst_width: width of destination image
    dst_height: height of destination image
    */

    //Raw copy of the source image (into the destination image)
    int x = get_global_id(0);
    int y = get_global_id(1);

    int src_pos;
    int dst_pos;
    
    for (int ix, iy; x < dst_width && y < dst_height; x++, y++) {
        ix = (int)(x * src_width / dst_width);
        iy = (int)(y * src_height / dst_height);
        src_pos = (int)(iy * src_width + ix);
        dst_pos = (int)(y * dst_width + x); //adapt to the upscaled image, so the source image is in the center
        
        dst[dst_pos] = src[src_pos];
    }

    //Upscaling
    const int max_dist = 10;
    const float weight[10] = {-10, -20, -30, -40, -50, -60, -70, -80, -90, -100}; 
    const float scale = (src_width / dst_width) * aa_factor;
    const int h_high = (int)(dst_height * scale);

    int idx = y * dst_width + x;

    if (x >= dst_width || y >= dst_height) return;

        idx = y * dst_width + x;
        uchar4 orig_color = dst[idx];

        float4 avg = convert_float4(orig_color);
        float total_weight = 1.0f;

        for (int dx = -max_dist; dx <= max_dist; dx++) {
            for (int dy = -max_dist; dy <= max_dist; dy++) {
                int nx = x + dx, ny = y + dy;
                if (nx >= 0 && nx < dst_width && ny >= 0 && ny < h_high) {
                    int n_idx = ny * dst_width + nx;
                    int d = (abs(dx) + abs(dy)) / 2;
                    float w = weight[d];
                    uchar4 neighbor = dst[n_idx];
                    avg += convert_float4(neighbor) * w;
                    total_weight += w;
                }
            }
        }

        avg /= total_weight;
        uchar4 new_color = convert_uchar4(clamp(avg, (float4)(0.0f), (float4)(255.0f)));
        dst[idx] = new_color; 
    

    //Downscaling (using the linear interpolation) --> Anti Aliasing
    const float downscale = aa_factor;
    const int h_low = (int)(dst_height / downscale);

    for (int dy = 0; dy < h_low; dy++) {
        for (int dx = 0; dx < dst_width; dx++) {
            int n_idx = dy * dst_width + dx;
            int idx = (int)(dy * downscale) * dst_width + (int)(dx * downscale);
            dst[idx] = dst[n_idx];
        }
    }
}