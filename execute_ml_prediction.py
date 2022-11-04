#newfile
import os
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import mat73
import sys
import matplotlib as mpl


def absolute_error(y_true, y_pred):
    error = y_pred - y_true
    abs_error = tf.math.abs(error)
    ########################################
    temp_y_pred = tf.expand_dims(y_pred, axis=-1)
    temp_y_pred = tf.image.resize(temp_y_pred, [32, 32], method="area")
    continuity_error = tf.Variable(tf.zeros([1, 32, 32, 1]))
    plot = False
    for i in range(32):
        left_value = 0.0
        left_disc_count = 0.0
        first_disc = 0.0
        last_disc = 0.0
        for j in range(32):
            if temp_y_pred[0][i, j][0] > left_value:
                left_value = temp_y_pred[0][i, j][0]
            if (left_value > .5) and (temp_y_pred[0][i, j-1] < .5):
                if first_disc == 0.0:
                    first_disc = j
                else:
                    last_disc = j
                left_disc_count+=1.0
        if left_disc_count > 1:
            continuity_error[0, i, first_disc:last_disc, 0].assign(tf.ones(last_disc-first_disc))
            #plot = True
    final_cont_error = tf.image.resize(continuity_error, [512, 512], method="bicubic")
    final_error = abs_error + final_cont_error[:, :, :, 0]
    if plot == True:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1), plt.imshow(abs_error[0])
        ax2 = fig.add_subplot(1, 3, 2), plt.imshow(final_cont_error[0, :, :, 0])
        ax3 = fig.add_subplot(1, 3, 3), plt.imshow(final_error[0])
        plt.show()

    return tf.reduce_mean(final_error)
    loss.__name__ = "absolute_error"


def load_model():
        val_saves = glob.glob('./*val_loss.h5')
        model = keras.models.load_model(val_saves[-1], custom_objects={'absolute_error': absolute_error})
        print("loaded model ", val_saves[-1])
        return model, val_saves[-1]


def predict(dataset, model):
    print("predicting...")
    predictions = []
    images = []

    img_set = dataset['lidar']
    img_set = np.where(np.isnan(img_set), 0, img_set)

    start_row = 0

    for i in range(int((len(img_set) / 1024)+1)):
        final_img_batch = np.zeros((512, 512, 4), dtype=np.float32)

        final_img_batch[:, :, 0] = cv2.resize(np.abs(img_set[start_row:int(start_row+1024), :, 0] / 31), (512, 512),
                                                 interpolation=cv2.INTER_AREA)
        final_img_batch[:, :, 1] = cv2.resize(img_set[start_row:int(start_row+1024), :, 1] / 7, (512, 512),
                                                 interpolation=cv2.INTER_AREA)
        final_img_batch[:, :, 2] = cv2.resize(img_set[start_row:int(start_row+1024), :, 2], (512, 512),
                                                 interpolation=cv2.INTER_AREA)
        final_img_batch[:, :, 3] = cv2.resize(img_set[start_row:int(start_row+1024), :, 3], (512, 512),
                                                 interpolation=cv2.INTER_AREA)
        final_img_batch = np.expand_dims(final_img_batch, axis=0)
        predictions.append(model[0].predict(final_img_batch, batch_size=10, verbose=0))
        images.append(final_img_batch)

        start_row += 1024

    predictions = np.asarray(predictions)
    images = np.asarray(images)
    predictions = np.squeeze(predictions)
    images = np.squeeze(images)

    # SIZE [WINDOWS, Y, X, CHANNELS]
    print("Reconstructing individual predictions...")
    temp_images = np.ndarray((len(images) * 512, 512, 4), dtype=np.float32)
    temp_predictions = np.ndarray((len(predictions) * 512, 512), dtype=np.float32)
    temp_images[:, :, :] = np.NaN
    temp_predictions[:, :] = np.NaN

    row = 0
    for i in range(len(images)):
        try:
            temp_images[row:(row + 512), :, :] = images[i, :, :]
        except ValueError:
            pass
        row += 512
    row = 0
    pass_no = 0
    for i in range(len(images)):
        """fig = plt.figure()
        ax0 = fig.add_subplot(1, 3, 1), plt.pcolormesh(temp_predictions[j, row:(row + 512), :])
        ax0[0].set_ylabel("Cross-shore distance")
        ax0[0].set_xlabel("Time")
        ax1 = fig.add_subplot(1, 3, 2), plt.pcolormesh(temp_images[j, row:(row + 512), :, 0])
        ax1[0].set_ylabel("Cross-shore distance")
        ax1[0].set_xlabel("Time")
        ax2 = fig.add_subplot(1, 3, 3), plt.pcolormesh(predictions[i, j, :, :])
        ax2[0].set_ylabel("Cross-shore distance")
        ax2[0].set_xlabel("Time")
        plt.show()"""
        try:
            temp_predictions[row:(row + 512), :] = \
                np.where(np.isnan(temp_predictions[row:(row + 512), :]), predictions[i, :, :],
                         (temp_predictions[row:(row + 512), :] * pass_no + predictions[i, :, :]) / (
                                 pass_no + 1))
        except ValueError:
            pass
            """temp_predictions[j, row:(row + 512), :] = np.where(np.isnan(temp_predictions[j, row:(row+512), :]),
                     predictions[i, j, -len(temp_predictions[j, row:]):, :],
                     (temp_predictions[j, row:(row+512), :]*pass_no + predictions[i, j, -len(temp_predictions[j, row:]):, :])/(pass_no+1))"""
        """fig = plt.figure()
        ax0 = fig.add_subplot(1, 3, 1), plt.pcolormesh(temp_predictions[j, row:(row + 512), :])
        ax0[0].set_ylabel("Cross-shore distance")
        ax0[0].set_xlabel("Time")
        ax1 = fig.add_subplot(1, 3, 2), plt.pcolormesh(temp_images[j, row:(row + 512), :, 0])
        ax1[0].set_ylabel("Cross-shore distance")
        ax1[0].set_xlabel("Time")
        ax2 = fig.add_subplot(1, 3, 3), plt.pcolormesh(predictions[i, j, :, :])
        ax2[0].set_ylabel("Cross-shore distance")
        ax2[0].set_xlabel("Time")
        plt.show()"""
        row += 512
    image = temp_images
    prediction = temp_predictions
    print("Images shape: ", np.shape(image))
    print("Predictions shape: ", np.shape(prediction))

    return image, prediction


def smooth(a, WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ
    r = np.arange(1, WSZ - 1, 2)
    start = np.cumsum(a[:WSZ - 1])[::2] / r
    stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def define_runupline(image, prediction):
        # defines runupline
        # saves to mat file
        print("Defining runup line...")

        image = cv2.resize(image[:int(original_matlength/2)], (1024, original_matlength), interpolation=cv2.INTER_CUBIC)
        prediction = cv2.resize(prediction[:int(original_matlength/2)], (1024, original_matlength), interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(image[:int(original_matlength/2), :, 2], (1024, original_matlength), interpolation=cv2.INTER_CUBIC)
        #image = np.where(image == 0, np.NaN, image)

        plotstart = 2048
        plotend = plotstart + 2048
        start = 200
        end = 800
        plot_distance = plotend - plotstart
        Y = np.linspace(0, plot_distance, plot_distance)#original_matlength, original_matlength)
        X = np.linspace(0, end - start, end - start)

        fig = plt.figure()
        fig.suptitle(file_location)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        ax0 = fig.add_subplot(1, 3, 2), plt.pcolormesh(prediction[plotstart:plotend, start:end])
        print(X)
        print(Y)
        cs = ax0[0].contour(X, Y, prediction[plotstart:plotend, start:end],
                            zorder=10,
                            vmin=0, vmax=1, alpha=1,
                            colors=['magenta'],
                            levels=[.5],
                            linestyles=['dashed'],
                            linewidths=[2, 2])
        ax0[0].set_title("Prediction Timestack")
        ax0[0].set_xlabel("Cross-shore distance (m)")
        #ax0[0].set_xticklabels(mpl.ticker.FormatStrFormatter('%.2i').format_ticks((ax0[0].get_xticks()) / 10))
        ax0[0].set_yticklabels(mpl.ticker.FormatStrFormatter('%.2i').format_ticks(ax0[0].get_yticks() / 7.1))
        ax0[0].invert_yaxis()
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax0[0])
        cbar.set_label("Water %")
        p = cs.collections[0].get_paths()
        list_len = [len(i) for i in p]
        longest_contour_index = np.argmax(np.array(list_len))
        v = p[longest_contour_index].vertices
        x = v[:, 0]
        y = v[:, 1]
        """ax1 = fig.add_subplot(1, 3, 2), plt.pcolormesh(image[plotstart:plotend, start:end, 2])
        ax1[0].set_title("Cumdiff")
        ax1[0].set_xlabel("Cross-shore distance (m)")
        ax1[0].set_xticklabels(mpl.ticker.FormatStrFormatter('%.2i').format_ticks((ax1[0].get_xticks() + start) / 10))
        cs = ax1[0].contour(X, Y, prediction[plotstart:plotend, start:end],
                            zorder=10,
                            vmin=0, vmax=1, alpha=1,
                            colors=['magenta'],
                            levels=[.5],
                            linestyles=['dashed'],
                            linewidths=[2, 2])"""
        norm = mpl.colors.Normalize(vmin=np.nanmin(image[plotstart:plotend, start:end, 0]), vmax=np.nanmax(image[plotstart:plotend, start:end, 0]))
        ax3 = fig.add_subplot(1, 3, 1), plt.pcolormesh(image[plotstart:plotend, start:end, 0], norm=norm)
        ax3[0].set_title("Reflectance Timestack")
        ax3[0].set_xlabel("Cross-shore distance (m)")
        ax3[0].set_ylabel("Time (s)")
        ax3[0].set_xticklabels(mpl.ticker.FormatStrFormatter('%.2i').format_ticks((ax3[0].get_xticks()) / 10))
        ax3[0].set_yticklabels(mpl.ticker.FormatStrFormatter('%.2i').format_ticks(ax3[0].get_yticks() / 7.1))
        ax3[0].invert_yaxis()
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax3[0])
        cbar.set_label("Intensity")
        cs = ax3[0].contour(X, Y, prediction[plotstart:plotend, start:end],
                            label='Beach-Water Interface',
                            zorder=10,
                            vmin=0, vmax=1, alpha=1,
                            colors=['magenta'],
                            levels=[.5],
                            linestyles=['dashed'],
                            linewidths=[2, 2])
        norm = mpl.colors.Normalize(vmin=0, vmax=.1)
        ax4 = fig.add_subplot(1, 3, 3), plt.pcolormesh(image[plotstart:plotend, start:end, 3]*7, norm=norm)
        ax4[0].set_title("Water Depth Timestack")
        ax4[0].set_xlabel("Cross-shore distance (m)")
        ax4[0].set_xticklabels(mpl.ticker.FormatStrFormatter('%.2i').format_ticks((ax4[0].get_xticks()) / 10))
        ax4[0].set_yticklabels(mpl.ticker.FormatStrFormatter('%.2i').format_ticks(ax4[0].get_yticks() / 7.1))
        ax4[0].invert_yaxis()
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax4[0])
        cbar.set_label("Water Depth (m)")
        cs = ax4[0].contour(X, Y, prediction[plotstart:plotend, start:end],
                            zorder=10,
                            vmin=0, vmax=1, alpha=1,
                            colors=['magenta'],
                            levels=[.5],
                            linestyles=['dashed'],
                            linewidths=[2, 2])
        plt.show()
        plt.close('all')

        downlineindex = []
        for j in range(len(image)):
            # find all indexes in x with value i
            # find all indexes in x with value i
            #   #average y values of those xes
            #   #return the average
            indexes = np.where((j + 1 > y) & (y >= j))

            if len(indexes[0]) > 0:
                avg_x = np.mean(x[indexes])
                downlineindex.append(avg_x)
            else:
                downlineindex.append(downlineindex[(j - 1)])

        downlineindex = np.array(downlineindex)

        return downlineindex, prediction


def return_classification(prediction, downlineindex):
        # saves classification to mat file
        print("Saving classification to mat file...")
        temp_prediction = np.ones((len(prediction), original_xshorelength))
        temp_prediction[:, 0:1024] = prediction
        plt.imshow(temp_prediction)
        plt.show()
        return {'beachWaterClassification': temp_prediction, 'downLineIndexML': downlineindex}


if __name__ == '__main__':
    file_location = sys.argv[1]
    MLInputs = mat73.loadmat(file_location)
    original_matlength = 0

    try:
        aGrid = MLInputs['aGridInterp']
        zGrid = MLInputs['zGridInterp']
        zDiffCumulative = MLInputs['zDiffCumulative']
    except:
        aGrid = MLInputs['lineGriddedData']['aGridInterp']
        zGrid = MLInputs['lineGriddedData']['zGridInterp']
        zDiffCumulative = MLInputs['lineRunupCalc']['zDiffCumulative']

    original_matlength = len(aGrid)
    original_xshorelength = len(aGrid[0])
    # calculate Zmin and Z-Zmin
    z_min = np.nanpercentile(zGrid, 95, axis=0)
    z_minus_zmin = np.zeros((len(zGrid), len(zGrid[0])))
    for i in range(len(zGrid)):
        z_minus_zmin[i, :] = zGrid[i, :] - z_min

    aGrid = aGrid[:, :1024]
    zGrid = zGrid[:, :1024]
    zDiffCumulative = zDiffCumulative[:, :1024]
    z_minus_zmin = z_minus_zmin[:, :1024]
    aGrid = np.expand_dims(aGrid, axis=-1)
    zGrid = np.expand_dims(zGrid, axis=-1)
    zDiffCumulative = np.expand_dims(zDiffCumulative, axis=-1)
    Z_minus_zmin = np.expand_dims(z_minus_zmin, axis=-1)

    lidar_obs = np.concatenate((aGrid, zGrid, zDiffCumulative, Z_minus_zmin), axis=-1)
    lidar_obs = np.where(np.isnan(lidar_obs), 0, lidar_obs)

    label_classes = zDiffCumulative
    label_classes = label_classes[:, :1024]
    label_classes = np.where(label_classes, 1, 0)

    dataset = {'lidar': lidar_obs, 'label': label_classes}

    test_model = load_model()
    image, prediction = predict(dataset, test_model)
    downlineindex, prediction = define_runupline(image, prediction)
    MLOutputs = return_classification(prediction, downlineindex)
    mat73.savemat('./temp_ML_outputs_prenamechange.mat', MLOutputs)
    os.rename('./temp_ML_outputs_prenamechange.mat', './temp_ML_outputs.mat')


