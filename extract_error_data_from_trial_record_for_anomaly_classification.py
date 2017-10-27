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
    anomalous_trial_folder = os.path.join(base_folder, "anomalous_trials")
    if not os.path.isdir(anomalous_trial_folder):
        raise Exception("anomalous trial folder not found")

    to_plot = []
    files = os.listdir(anomalous_trial_folder)
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

        to_plot.append([
            f,
            tag_multimodal_df,
            list_of_anomaly_start_time,
        ])

    dimensions = [
        '.endpoint_state.pose.position.x',
        '.endpoint_state.pose.position.y',
        '.endpoint_state.pose.position.z',
        '.endpoint_state.pose.orientation.x',
        '.endpoint_state.pose.orientation.y',
        '.endpoint_state.pose.orientation.z',
        '.endpoint_state.pose.orientation.w',
        '.wrench_stamped.wrench.force.x',
        '.wrench_stamped.wrench.force.y',
        '.wrench_stamped.wrench.force.z',
        '.wrench_stamped.wrench.torque.x',
        '.wrench_stamped.wrench.torque.y',
        '.wrench_stamped.wrench.torque.z',
    ]

    import datetime
    output_dir = os.path.join(base_folder, "extracted_anomalies", str(datetime.datetime.now()))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    visualization_by_dimension_dir = os.path.join(output_dir, 'visualization_by_dimension')
    os.makedirs(visualization_by_dimension_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for dim in dimensions:
        subplot_amount = len(to_plot)
        fig, axs = plt.subplots(nrows=subplot_amount, ncols=1, sharex=True, sharey=True)
        if subplot_amount == 1:
            axs = [axs]

        for idx, tmp in enumerate(to_plot):
            f, tag_multimodal_df, list_of_anomaly_start_time = tmp
            df = tag_multimodal_df
            if dim not in df:
                continue

            ax = axs[idx]
            ax.plot(
                df['time'].tolist(),
                df[dim].tolist(), 
            )
            ax.set_title('trial: '+f+'.bag')
            color_bg_and_anomaly(
                ax,
                df,
                list_of_anomaly_start_time,
            )
        fig.set_size_inches(16,4*subplot_amount)
        fig.suptitle(dim)
        fig.savefig(os.path.join(visualization_by_dimension_dir, dim+'.png'))
