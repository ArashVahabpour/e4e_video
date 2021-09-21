def create_csv(data_path):
    import os
    import glob
    import csv

    csv_filename = 'data/train.csv'
    #if os.path.exists(csv_filename):
    #    return

    with open(csv_filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for filename in glob.glob(os.path.join(data_path, '**', '*.mp4'), recursive=True):
            writer.writerow([filename, 0])

data_path = 'data/'#Actor_01'
create_csv(data_path)
