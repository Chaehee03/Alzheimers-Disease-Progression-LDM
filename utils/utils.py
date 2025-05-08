# def tb_display_generation(writer, step, tag, image):
#     """
#     Display generation result in TensorBoard during Diffusion Model training.
#     """
#     plt.style.use('dark_background')
#     _, ax = plt.subplots(ncols=3, figsize=(7, 3))
#     for _ax in ax.flatten(): _ax.set_axis_off()
#
#     ax[0].imshow(image[image.shape[0] // 2, :, :], cmap='gray')
#     ax[1].imshow(image[:, image.shape[1] // 2, :], cmap='gray')
#     ax[2].imshow(image[:, :, image.shape[2] // 2], cmap='gray')
#
#     plt.tight_layout()
#     writer.add_figure(tag, plt.gcf(), global_step=step)