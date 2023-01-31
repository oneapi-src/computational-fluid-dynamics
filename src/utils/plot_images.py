# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
This script includes utility functions
for plotting fluid flow images
'''

import os
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pathlib

def plot_image_single(boundary_arr, predicted_flow, filename):
    
    filename = filename.replace('.png', '_flow.png')
    
    value = np.sqrt(np.square(predicted_flow[0, :, :, 0]) + np.square(predicted_flow[0, :, :, 1]))
    sflow_plot = value - .05 * boundary_arr[0, :, :, 0]
    print(value.shape, predicted_flow.shape, boundary_arr.shape)
    
    fig, ax = plt.subplots()
    im = ax.imshow(sflow_plot, cmap='inferno')
    im.set_clim(vmin=-0.05, vmax=0.3)
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_images(predicted_flow, boundary_batch, label_batch, batch_size, start_ctr):
    image_dir = "./output_images/"
    path = pathlib.Path(image_dir)
    path.mkdir(parents=True, exist_ok=True)
    for counter in range(batch_size):
        file_index = start_ctr+counter
        vel = {}
        
        vel['actual'] = np.sqrt(np.square(label_batch[counter, :, :, 0]) + np.square(label_batch[counter, :, :, 1]))
        vel['pred'] = np.sqrt(np.square(predicted_flow[counter, :, :, 0]) + np.square(predicted_flow[counter, :, :, 1]))
        vel['diff'] = np.absolute(vel['actual']-vel['pred'])
        file_list = []
        for key, value in vel.items():
            filename = "./output_images/" + str(file_index) + "_" + key + ".png"
            sflow_plot = value - .05 * boundary_batch[counter, :, :, 0]
            fig, ax = plt.subplots()
            im = ax.imshow(sflow_plot, cmap='inferno')
            im.set_clim(vmin=-0.05, vmax=0.3)
            ax.axis('off')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)
            fig.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            file_list.append(filename)
            
        images = [Image.open(x) for x in file_list]
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        full_file_name = "./output_images/" + str(file_index) + "_ALL" + ".png"
        new_im.save(full_file_name)
        for image in file_list:
            os.remove(image)
