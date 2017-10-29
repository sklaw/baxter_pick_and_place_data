import os
import pandas as pd
from datetime import datetime, timedelta
from mpl_toolkits.mplot3d import Axes3D
import load_data_folder
import plot_data_in_panda_df
import matplotlib.pyplot as plt
import ipdb
import matplotlib.dates as mdates 
import numpy as np
from matplotlib.pyplot import cm 
import copy
import birl.robot_introspection_pkg.multi_modal_config as mmc

PLOT_VERIFICATION = True 

def trim_non_trial_data(tag_multimodal_df, hmm_online_result_df):
    state_df = tag_multimodal_df[tag_multimodal_df['.tag'] != 0]
    row = state_df.head(1)
    trial_start_time = row['time'].values[0]
    row = state_df.tail(1)
    trial_end_time = row['time'].values[0]

    return tag_multimodal_df[(tag_multimodal_df['time']>=trial_start_time) & (tag_multimodal_df['time']<=trial_end_time)], \
        hmm_online_result_df[(hmm_online_result_df['time']>=trial_start_time) & (hmm_online_result_df['time']<=trial_end_time)]

def color_bg_and_anomaly(
    plot,
    tag_df,
    list_of_anomaly_start_time,
):
    tag_df_length = tag_df.shape[0]
    start_t = 0
    state_color = {0: "gray", 2: "green", 5: "green"}
    color=iter(cm.rainbow(np.linspace(0, 1, 10)))
    for t in range(1, tag_df_length):
        if tag_df['.tag'][t-1] == tag_df['.tag'][t] and t < len(tag_df['.tag'])-1:
            continue
        skill = tag_df['.tag'][t-1]
        end_t = t
        if skill == -1:
            color = 'red'
        elif skill == -2:
            color = 'black'
        elif skill == -3:
            color = 'yellow'
        else:
            color = state_color[skill]
        plot.axvspan(tag_df['time'][start_t], tag_df['time'][end_t], facecolor=color, ymax=1, ymin=0.95)
        start_t = t

    for t in list_of_anomaly_start_time:
        plot.axvline(t, color='red')
        plot.axvspan(t-2, t+2, facecolor='pink', ymax=0.95, ymin=0)



def get_anomaly_range(tag_df, flag_df):
    list_of_anomaly_start_time = [flag_df['time'][0]]
    
    flag_df_length = flag_df.shape[0]
    for idx in range(1, flag_df_length):
        now_time = flag_df['time'][idx]
        last_time = flag_df['time'][idx-1]
        if now_time-last_time > timedelta(seconds=2):
            list_of_anomaly_start_time.append(now_time) 

    return list_of_anomaly_start_time

if __name__ == "__main__":
    from optparse import OptionParser
    usage = "usage: %prog -d base_folder_path"
    parser = OptionParser(usage=usage)

    parser.add_option("-d", "--base-folder",
        action="store", type="string", dest="base_folder",
        help="provide a base folder which will have this structure: ./01, ./01/*.csv, ./02, ./02/*.csv, ...")
    (options, args) = parser.parse_args()

    if options.base_folder is None:
        parser.error("no base_folder")

    base_folder = options.base_folder

    anomalous_trial_folder = os.path.join(base_folder, "anomalous_trial_rosbags")
    if not os.path.isdir(anomalous_trial_folder):
        raise Exception("anomalous trial folder not found")



    import datetime
    extracted_anomalies_dir = os.path.join(base_folder, "extracted_anomalies_dir", str(datetime.datetime.now()))
    os.makedirs(extracted_anomalies_dir)
    raw_anomalies_dir = os.path.join(extracted_anomalies_dir, 'raw_anomalies_dir')
    os.makedirs(raw_anomalies_dir)
    resampled_anomalies_dir = os.path.join(extracted_anomalies_dir, 'resampled_anomalies_dir')
    os.makedirs(resampled_anomalies_dir)

    interested_data_fields = copy.deepcopy(mmc.interested_data_fields)
    interested_data_fields.append('time')

    to_plot = []
    files = os.listdir(anomalous_trial_folder)
    files.sort()
    for f in files:
        path = os.path.join(anomalous_trial_folder, f)
        if not os.path.isdir(path):
            continue
        print 'processing', f

        if os.path.isfile(os.path.join(path, f+'-tag_multimodal.csv')):
            tag_multimodal_csv_path = os.path.join(path, f+'-tag_multimodal.csv')
        elif os.path.isfile(os.path.join(path, 'tag_multimodal.csv')):
            tag_multimodal_csv_path = os.path.join(path, 'tag_multimodal.csv')
        else:
            raise Exception("folder %s doesn't have tag_multimodal csv file."%(path,))

        if os.path.isfile(os.path.join(path, f+'-anomaly_detection_signal.csv')):
            hmm_online_result_csv_path = os.path.join(path, f+'-anomaly_detection_signal.csv')
        else:
            raise Exception("folder %s doesn't have hmm_online_result csv file."%(path,))

        tag_multimodal_df = pd.read_csv(tag_multimodal_csv_path, sep=',')
        tag_multimodal_df = tag_multimodal_df[interested_data_fields]
        hmm_online_result_df = pd.read_csv(hmm_online_result_csv_path, sep=',')


        from dateutil import parser
        tag_multimodal_df['time'] = tag_multimodal_df['time'].apply(lambda x: parser.parse(x))
        hmm_online_result_df['time'] = hmm_online_result_df['time'].apply(lambda x: parser.parse(x))
        tag_multimodal_df, hmm_online_result_df = trim_non_trial_data(tag_multimodal_df, hmm_online_result_df)
        tag_multimodal_df.index = np.arange(len(tag_multimodal_df))
        hmm_online_result_df.index = np.arange(len(hmm_online_result_df))

        list_of_anomaly_start_time = get_anomaly_range(
            tag_multimodal_df,
            hmm_online_result_df,
        )

        start_t = tag_multimodal_df['time'][0]
        tag_multimodal_df['time'] -= start_t
        tag_multimodal_df['time'] = tag_multimodal_df['time'].apply(lambda x: x.total_seconds())
        for i in range(len(list_of_anomaly_start_time)):
            list_of_anomaly_start_time[i] -= start_t
            list_of_anomaly_start_time[i] = list_of_anomaly_start_time[i].total_seconds()

        from birl.robot_introspection_pkg.anomaly_sampling_config import anomaly_window_size_in_sec, anomaly_resample_hz
        list_of_anomaly_df = []
        list_of_resampled_anomaly_df = []
        for anomaly_idx, anomaly_t in enumerate(list_of_anomaly_start_time):
            search_start = anomaly_t-anomaly_window_size_in_sec/2
            search_end = anomaly_t+anomaly_window_size_in_sec/2
            anomaly_df = tag_multimodal_df[\
                (tag_multimodal_df['time'] >= search_start) &\
                (tag_multimodal_df['time'] <= search_end)\
            ]
            anomaly_df = anomaly_df.drop('.tag', axis=1).set_index('time')
            anomaly_name = 'no_%s_from_trial_%s'%(anomaly_idx, f)
            anomaly_df.to_csv(os.path.join(raw_anomalies_dir, anomaly_name+'.csv'))
            list_of_anomaly_df.append(anomaly_df)

            search_start = anomaly_t-anomaly_window_size_in_sec/2-1            
            search_end = anomaly_t+anomaly_window_size_in_sec/2+1
            search_df = tag_multimodal_df[\
                (tag_multimodal_df['time'] >= search_start) &\
                (tag_multimodal_df['time'] <= search_end)\
            ]
            search_df = search_df.drop('.tag', axis=1).set_index('time')
            new_time_index = np.linspace(anomaly_t-anomaly_window_size_in_sec/2, anomaly_t+anomaly_window_size_in_sec/2, anomaly_window_size_in_sec*anomaly_resample_hz)
            old_time_index = search_df.index
            resampled_anomaly_df = search_df.reindex(old_time_index.union(new_time_index)).interpolate(method='linear', axis=0).ix[new_time_index]
            anomaly_name = 'resampled_%shz_no_%s_from_trial_%s'%(anomaly_resample_hz, anomaly_idx, f)
            resampled_anomaly_df.to_csv(os.path.join(resampled_anomalies_dir, anomaly_name+'.csv'))
            list_of_resampled_anomaly_df.append(resampled_anomaly_df)

        to_plot.append([
            f,
            tag_multimodal_df,
            list_of_anomaly_start_time,
            list_of_anomaly_df,
            list_of_resampled_anomaly_df,
        ])

    if not PLOT_VERIFICATION:
        import sys
        sys.exit(0)
        

    dimensions = copy.deepcopy(mmc.interested_data_fields)
    if '.tag' in dimensions:
        idx_to_del = dimensions.index('.tag')
        del dimensions[idx_to_del]

    visualization_by_dimension_dir = os.path.join(extracted_anomalies_dir, 'visualization_by_dimension_dir')
    os.makedirs(visualization_by_dimension_dir)
    anomaly_in_trial_dir = os.path.join(visualization_by_dimension_dir, "anomaly_in_trial_dir")
    os.makedirs(anomaly_in_trial_dir)
    anomaly_by_trial_dir = os.path.join(visualization_by_dimension_dir, "anomaly_by_trial_dir")
    os.makedirs(anomaly_by_trial_dir)

    for dim in dimensions:
        trial_amount = len(to_plot)
        anomaly_in_trial_fig, axs_for_ait_fig = plt.subplots(nrows=trial_amount, ncols=1, sharex=True, sharey=True)
        if trial_amount == 1:
            axs_for_ait_fig = [axs_for_ait_fig]

        for idx, tmp in enumerate(to_plot):
            f, \
            tag_multimodal_df, \
            list_of_anomaly_start_time, \
            list_of_anomaly_df, \
            list_of_resampled_anomaly_df = tmp

            ax = axs_for_ait_fig[idx]
            ax.plot(
                tag_multimodal_df['time'].tolist(),
                tag_multimodal_df[dim].tolist(), 
            )
            ax.set_title('trial: '+f+'.bag')
            color_bg_and_anomaly(
                ax,
                tag_multimodal_df,
                list_of_anomaly_start_time,
            )

            for anomaly_df in list_of_anomaly_df:
                ax.plot(
                    anomaly_df.index.tolist(),
                    anomaly_df[dim].tolist(),
                    color='red',
                )

            trial_dir = os.path.join(anomaly_by_trial_dir, f)
            if not os.path.isdir(trial_dir):
                os.makedirs(trial_dir)
    
            anomaly_amount = len(list_of_anomaly_df)
            anomalies_in_this_trial_fig, axs_for_aitt_fig = plt.subplots(nrows=anomaly_amount, ncols=2, sharex=True, sharey=True)
            if anomaly_amount == 1:
                axs_for_aitt_fig.reshape(1, 2)

            for anomaly_idx in range(len(list_of_anomaly_df)):
                anomaly_df = list_of_anomaly_df[anomaly_idx]
                ax = axs_for_aitt_fig[anomaly_idx, 0] 
                time_x = anomaly_df.index-anomaly_df.index[0]
                ax.plot(
                    time_x.tolist(),
                    anomaly_df[dim].tolist(), 
                )
                anomaly_name = 'no_%s_from_trial_%s'%(anomaly_idx, f)
                ax.set_title(anomaly_name)

                resampled_anomaly_df = list_of_resampled_anomaly_df[anomaly_idx]
                ax = axs_for_aitt_fig[anomaly_idx, 1] 
                time_x = resampled_anomaly_df.index-resampled_anomaly_df.index[0]
                ax.plot(
                    time_x.tolist(),
                    resampled_anomaly_df[dim].tolist(), 
                )
                anomaly_name = 'resampled_%shz_no_%s_from_trial_%s'%(anomaly_resample_hz, anomaly_idx, f)
                ax.set_title(anomaly_name)
            anomalies_in_this_trial_fig.set_size_inches(16,4*anomaly_amount)
            anomalies_in_this_trial_fig.suptitle(f+' '+dim)
            anomalies_in_this_trial_fig.savefig(os.path.join(trial_dir, 'trial_'+f+'_'+dim+'.png'))
            plt.close(anomalies_in_this_trial_fig)
                    
        anomaly_in_trial_fig.set_size_inches(16,4*trial_amount)
        anomaly_in_trial_fig.suptitle(dim)
        anomaly_in_trial_fig.savefig(os.path.join(anomaly_in_trial_dir, dim+'.png'))
        plt.close(anomaly_in_trial_fig)


