"""
Smart Upscaling 2
--> Smart Upscaling 4xKNN1

This is a script for Smart Upscaling 4xKNN1.
The OpenCL kernel is stored in KNN1/kernel1.c

It uses a similar method as Smart Upscaling 1, but divises the image into 4 blocs (performance mode).

Modes : 
- Performance : 4 blocks* 
- Balanced : 8 blocks*
- Quality : 16 blocks*

- Ultra Performance : 1 block
- Ultra Balanced : n blocks (square shape, AI decided)
- Ultra Quality : n blocks (variable shape, AI decided)

*for a Full HD / 1080p image
"""
import pyopencl as cl
import numpy as np
from PIL import Image


def upscale_Ultra_Performance(image, scale):
    # Define the anti-aliasing factor
    anti_aliasing = 256

    # Load the low resolution image and convert to RGBA
    PIL_lowres = Image.open(image).convert("RGBA")
    lowres = np.asarray(PIL_lowres).astype(np.uint8)

    # Create the high resolution image array (writable)
    new_width = int(PIL_lowres.width * scale)
    new_height = int(PIL_lowres.height * scale)
    highres = np.zeros((new_height, new_width, 4), dtype=np.uint8)

    # Read the kernel code
    with open("KNN1/kernel1.c", "r") as cl_kernel:
        kernel_code = cl_kernel.read()

    # Create the platform and device
    platforms = cl.get_platforms()
    device = platforms[0].get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    # Create the buffers
    mf = cl.mem_flags
    buffer_lowres = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lowres)
    buffer_highres = cl.Buffer(context, mf.WRITE_ONLY, highres.nbytes)

    # Build the program
    try:
        program = cl.Program(context, kernel_code).build()
    except cl.RuntimeError as e:
        print("Build error:")
        print(e)
        exit(1)

    # Set kernel arguments and execute
    kernel = program.upscale
    kernel.set_args(buffer_lowres, buffer_highres, 
                np.float32(lowres.shape[1]),  # width
                np.float32(lowres.shape[0]),  # height
                np.float32(highres.shape[1]), # width
                np.float32(highres.shape[0]),  # height
                np.float32(anti_aliasing),
                    )
    global_size = (highres.shape[1], highres.shape[0])
    cl.enqueue_nd_range_kernel(queue, kernel, global_size, None)

    # Copy results back to host
    cl.enqueue_copy(queue, highres, buffer_highres).wait()

    # Save the high resolution image (convert back to RGB)
    img=Image.fromarray(highres, 'RGBA').convert('RGB')
    img.save(f"tests/result_UltraPerformance_x{scale}.jpg")
    img.show()



if __name__ == "__main__":
    upscale_Ultra_Performance("tests/image.jpg", 2)