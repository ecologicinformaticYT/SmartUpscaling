import pyopencl as cl
import numpy as np
from PIL import Image

def KNeuralNetwork():
    # Charger les images
    lowres_img = Image.open("./images/image.png").convert("RGBA")

    lowres = np.array(lowres_img, dtype=np.uint8)

    h_low, w_low, _ = lowres.shape
    scale = 2 # Facteur d'upscale
    h_high, w_high = h_low * scale, w_low * scale

    # Image upscalée (copie brute des pixels)
    highres = np.zeros((h_high, w_high, 4), dtype=np.uint8)
    for x in range(w_low):
        for y in range(h_low):
            highres[y * scale:(y + 1) * scale, x * scale:(x + 1) * scale] = lowres[y, x]

    # Initialisation OpenCL
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    highres_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=highres)

    # Initialisation des poids pour les voisins
    weight = np.array([-5.5483876e+42])
    #retenue (temp) : Poids après itération 50: [-4.41041425e+42]
    weight_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=weight)

    # Kernel OpenCL avec ajustement des poids
    kernel_code = """
    __kernel void knn_upscale(__global uchar4 *highres, __global float *weight,
                              int w_high, int h_high, int scale, int max_dist) {
        int x = get_global_id(0);
        int y = get_global_id(1);

        if (x >= w_high || y >= h_high) return;

        int idx = y * w_high + x;
        uchar4 orig_color = highres[idx];

        float4 avg = convert_float4(orig_color);
        float total_weight = 1.0f;

        for (int dx = -max_dist; dx <= max_dist; dx++) {
            for (int dy = -max_dist; dy <= max_dist; dy++) {
                int nx = x + dx, ny = y + dy;
                if (nx >= 0 && nx < w_high && ny >= 0 && ny < h_high) {
                    int n_idx = ny * w_high + nx;
                    int d = (abs(dx) + abs(dy)) / 2;
                    float w = weight[d];
                    uchar4 neighbor = highres[n_idx];
                    avg += convert_float4(neighbor) * w;
                    total_weight += w;
                }
            }
        }

        avg /= total_weight;
        uchar4 new_color = convert_uchar4(clamp(avg, (float4)(0.0f), (float4)(255.0f)));
        highres[idx] = new_color;
    }
    """

    # Compilation du programme
    prg = cl.Program(ctx, kernel_code).build()

    # Phase d'entraînement
    max_dist = 10
    prg.knn_upscale(queue, (w_high, h_high), None, highres_buf, weight_buf, np.int32(w_high), np.int32(h_high), np.int32(scale), np.int32(max_dist))

    # Récupération de l'image finale
    cl.enqueue_copy(queue, highres, highres_buf).wait()

    # Sauvegarde et affichage
    final_img = Image.fromarray(highres, "RGBA")
    final_img.save("./images/res_final2x.png")
    final_img.show()

    print("✅ Image finale générée avec KNN.")
    
KNeuralNetwork()