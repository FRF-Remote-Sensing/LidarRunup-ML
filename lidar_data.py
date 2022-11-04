import settings
from torch.utils.data.dataset import Dataset
import glob
import numpy as np
import mat73
from scipy.io import loadmat, savemat
import matplotlib as mpl
import matplotlib.pyplot as plt

if __name__ == '__main__':
    linedatamat = glob.glob('F:/DuneLidar/QAQC0/*.mat')
    linedatamat = sorted(linedatamat)
    print(linedatamat)
    npydata = glob.glob('./data/features_labels*')

    print("loading all data...")
    for i in range(len(linedatamat)):
        linedataname = linedatamat[i][19:-23]
        already_Exists = [i for i in npydata if linedataname in i]
        """if len(already_Exists) > 0:
            print("npyFile " + linedataname + " already exists")
            continue"""
        print(linedataname)
        try:
            try:
                matfile = mat73.loadmat(linedatamat[i])
            except:
                matfile = loadmat(linedatamat[i])

            QAQC = matfile['lineRunupCalc']['isQAQC']
            print("isQAQC: ", QAQC)
            #if QAQC == 1:
            lineGriddedData = matfile['lineGriddedData']
            lineRunupCalc = matfile['lineRunupCalc']

            # assign variables to write to pickle file
            label_classes = lineRunupCalc['griddedDataIsWater']
            label_classes = np.where(label_classes, 1, 0)

            aGrid = lineGriddedData['aGridInterp']
            zGrid = lineGriddedData['zGridInterp']

            zGrid = np.where(label_classes == 0, 0, zGrid)

            zDiffCumulative = lineRunupCalc['zDiffCumulative']
            average_swash = int(np.amax(lineRunupCalc['downLineIndex']))
            VWG80 = 373
            VWG90 = 473
            VWG100 = 573
            VWG110 = 673
            VWG140 = 973

            average_swash_nan = np.count_nonzero(np.isnan(zGrid[:, average_swash]))
            vwg80_nans = np.count_nonzero(np.isnan(zGrid[:, VWG80]))
            vwg90_nans = np.count_nonzero(np.isnan(zGrid[:, VWG90]))
            vwg100_nans = np.count_nonzero(np.isnan(zGrid[:, VWG100]))
            vwg110_nans = np.count_nonzero(np.isnan(zGrid[:, VWG110]))
            vwg140_nans = np.count_nonzero(np.isnan(zGrid[:, VWG140]))

            waterswash_len = len(np.where(label_classes[:, average_swash] == 1)[0]) + 1
            water80_len = len(np.where(label_classes[:, VWG80] == 1)[0]) + 1
            water90_len = len(np.where(label_classes[:, VWG90] == 1)[0]) + 1
            water100_len = len(np.where(label_classes[:, VWG100] == 1)[0]) + 1
            water110_len = len(np.where(label_classes[:, VWG110] == 1)[0]) + 1
            water140_len = len(np.where(label_classes[:, VWG140] == 1)[0]) + 1

            print("waterswash_nans: ", average_swash_nan)
            print("waterswash_length: ", waterswash_len)
            print("vwg80_nans: ", vwg80_nans)
            print("water80_len: ", water80_len)
            print("vwg90_nans: ", vwg90_nans)
            print("water90_len: ", water90_len)
            print("vwg100_nans: ", vwg100_nans)
            print("water100_len: ", water100_len)
            print("vwg110_nans: ", vwg110_nans)
            print("water110_len: ", water110_len)
            print("vwg140_nans: ", vwg140_nans)
            print("water140_len: ", water140_len)
            print("vwg80 nan %: ", vwg80_nans / water80_len)
            print("vwg90 nan %: ", vwg90_nans / water90_len)
            print("vwg100 nan %: ", vwg100_nans / water100_len)
            print("vwg110 nan %: ", vwg110_nans / water110_len)
            print("vwg140 nan %: ", vwg140_nans / water140_len)
            print("___________________________")
            file = open('QAQC0nancounts.txt', 'a')
            file.write(linedataname + " " +
                        str(average_swash_nan / waterswash_len) + " " +
                        str(vwg80_nans / water80_len) + " " +
                        str(vwg90_nans / water90_len) + " " +
                        str(vwg100_nans / water100_len) + " " +
                        str(vwg110_nans / water110_len) + " " +
                        str(vwg140_nans / water140_len) +
                       '\n')
            file.close()
            # calculate Zmin and Z-Zmin
            z_min = np.nanpercentile(zGrid, 95, axis=0)
            z_minus_zmin = np.zeros((len(zGrid), len(zGrid[0])))
            for i in range(len(zGrid)):
                z_minus_zmin[i, :] = zGrid[i, :] - z_min

            aGrid = aGrid[:, :settings.dimensions]
            zGrid = zGrid[:, :settings.dimensions]
            zDiffCumulative = zDiffCumulative[:, :settings.dimensions]
            z_minus_zmin = z_minus_zmin[:, :settings.dimensions]
            label_classes = label_classes[:, :settings.dimensions]
            aGrid = np.expand_dims(aGrid, axis=-1)
            zGrid = np.expand_dims(zGrid, axis=-1)
            zDiffCumulative = np.expand_dims(zDiffCumulative, axis=-1)
            Z_minus_zmin = np.expand_dims(z_minus_zmin, axis=-1)

            lidar_obs = np.concatenate((aGrid, zGrid, zDiffCumulative, Z_minus_zmin), axis=-1)
            lidar_obs = np.where(np.isnan(lidar_obs), 0, lidar_obs)

            sample = {'lidar': lidar_obs, 'label': label_classes}#, "r2": r2}
            np.save('./data/features_labels' + linedataname + '.npy', sample)
        except:
            print("read length non negative -1 error when using scipy loadmat")


class LidarDataset(Dataset):
    def __init__(self, transform=None):
        self.data_list = glob.glob('./data/features_labels*.npy')
        self.test_list = glob.glob('./data/test/features_labels*.npy')

    def __getitem__(self, idx):
        return self.load_file(idx)

    def __len__(self):
        return len(self.data_list)

    def load_file(self, idx):
        self.test_list.sort(key=settings.natural_keys)
        self.data_list.sort(key=settings.natural_keys)

        if idx >= 10000:
            idx = idx - 10000
            sample = np.load(self.test_list[idx], allow_pickle=True)
            startTime = settings.start_row
            endTime = startTime + settings.dimensions
        else:
            sample = np.load(self.data_list[idx], allow_pickle=True)
            startTime = np.random.randint(0, settings.matlength- settings.dimensions)

            if startTime > (settings.matlength - settings.dimensions):
                startTime = (settings.matlength - (settings.dimensions+50))
            endTime = startTime + settings.dimensions

        sample[()]['lidar'] = sample[()]['lidar'][startTime:endTime, :, :]
        sample[()]['label'] = sample[()]['label'][startTime:endTime, :]
        if settings.start_row > 6144:
            """print(self.data_list[idx])
            print(startTime)
            image = sample[()]['lidar']
            label = sample[()]['label']
            fig = plt.figure()
            ax0 = fig.add_subplot(2, 3, 1), plt.imshow(image[:, :, 0])
            ax0[0].set_xlabel("Cross-shore distance")
            ax0[0].set_ylabel("Time")
            ax1 = fig.add_subplot(2, 3, 2), plt.imshow(image[:, :, 1])
            ax1[0].set_xlabel("Cross-shore distance")
            ax2 = fig.add_subplot(2, 3, 3), plt.imshow(label)
            ax2[0].set_ylabel("Cross-shore distance")
            ax2[0].set_xlabel("Time")
            ax3 = fig.add_subplot(2, 3, 4), plt.imshow(image[:, :, 2])
            ax3[0].set_xlabel("Cross-shore distance")
            ax3[0].set_ylabel("Time")
            ax4 = fig.add_subplot(2, 3, 5), plt.imshow(image[:, :, 3])
            ax4[0].set_xlabel("Cross-shore distance")
            ax5 = fig.add_subplot(2, 3, 6), plt.imshow(label)
            ax5[0].set_ylabel("Cross-shore distance")
            ax5[0].set_xlabel("Time")
            plt.tight_layout()
            plt.show()"""
        return sample
