import os
import pandas as pd
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import load_data_folder
import plot_data_in_panda_df
import matplotlib.pyplot as plt
import ipdb

def datetime_to_float(d):
    epoch = datetime.utcfromtimestamp(0)
    total_seconds =  (d - epoch).total_seconds()
    # total_seconds will be in decimals (millisecond precision)
    return total_seconds

def get_error_range(tag_pd, flag_pd):
    list_of_range = []
    
    flag_start_idx = 0
    tag_start_idx = 0

    flag_pd_length = flag_pd.shape[0]
    tag_pd_length = tag_pd.shape[0]

    anomaly_start_time = None
    anomaly_end_time = flag_pd['time'][0]
    anomaly_detected = False
    while True:
        if not anomaly_detected:
            tmp = flag_pd.query('time > %s'%(anomaly_end_time,)).head(1).index
            if len(tmp) == 0:
                break
            flag_start_idx = tmp[0] 
            for idx in range(flag_start_idx, flag_pd_length):
                if flag_pd['.event_flag'][idx] == 0:
                    backtracked_idx = idx 
                    while flag_pd['.event_flag'][backtracked_idx] <= flag_pd['.event_flag'][idx]:
                        backtracked_idx -= 1

                    anomaly_detected = True
                    anomaly_start_time = flag_pd['time'][backtracked_idx]
                    break
            if not anomaly_detected:
                break
        else:
            tmp = tag_pd.query('time > %s'%(anomaly_start_time,)).head(1).index
            if len(tmp) == 0:
                break
            tag_start_idx = tmp[0] 
            for idx in range(tag_start_idx, tag_pd_length):
                if tag_pd['.tag'][idx] == 0:
                    anomaly_detected = False 
                    anomaly_end_time = tag_pd['time'][idx]
                    list_of_range.append([
                        anomaly_start_time,
                        anomaly_end_time,
                    ])
                    break
            if anomaly_detected:
                break
    return list_of_range

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

    files = os.listdir(base_folder)
    for f in files:
        path = os.path.join(base_folder, f)
        if not os.path.isdir(path):
            continue

        if os.path.isfile(os.path.join(path, f+'-tag_multimodal.csv')):
            tag_multimodal_csv_path = os.path.join(path, f+'-tag_multimodal.csv')
        elif os.path.isfile(os.path.join(path, 'tag_multimodal.csv')):
            tag_multimodal_csv_path = os.path.join(path, 'tag_multimodal.csv')
        else:
            raise Exception("folder %s doesn't have tag_multimodal csv file."%(path,))

        if os.path.isfile(os.path.join(path, f+'-hmm_online_result.csv')):
            hmm_online_result_csv_path = os.path.join(path, f+'-hmm_online_result.csv')
        elif os.path.isfile(os.path.join(path, 'hmm_online_result.csv')):
            hmm_online_result_csv_path = os.path.join(path, 'hmm_online_result.csv')
        else:
            raise Exception("folder %s doesn't have hmm_online_result csv file."%(path,))

        tag_multimodal_pd = pd.read_csv(tag_multimodal_csv_path, sep=',')
        hmm_online_result_pd = pd.read_csv(hmm_online_result_csv_path, sep=',')

        from dateutil import parser
        tag_multimodal_pd['time'] = tag_multimodal_pd['time'].apply(lambda x: datetime_to_float(parser.parse(x)))
        hmm_online_result_pd['time'] = hmm_online_result_pd['time'].apply(lambda x: datetime_to_float(parser.parse(x)))

        list_of_range = get_error_range(
            tag_multimodal_pd,
            hmm_online_result_pd,
        )
   
        print list_of_range 

        fig = plt.figure()
        deri_of_diff = fig.add_subplot(111)
        deri_of_diff.plot(
            hmm_online_result_pd['time'].tolist(),
            hmm_online_result_pd['.deri_of_diff_btw_curlog_n_thresh.data'].tolist(),
            marker='o',
        )
        for idx, r in enumerate(list_of_range):
            deri_of_diff.axvspan(r[0], r[1], facecolor='red', alpha=0.25)
            tag_multimodal_pd[(tag_multimodal_pd['time']>=r[0]) & (tag_multimodal_pd['time']<=r[1])].to_csv(os.path.join(path, 'extracted_error_%s.csv'%(idx,)))            
        title = 'error range of trial %s'%(f,)
        deri_of_diff.set_title(title)
        fig.savefig(os.path.join(path, title+".png"), format="png", dpi=900)



