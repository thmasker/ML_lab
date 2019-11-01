

import pandas as pd


df = pd.read_csv("T2_tuesdays.csv")
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
#print('\n'.join(df.columns.to_list()))

'''with open('head.txt', 'w') as out:
    out.write(
        df[[column for column in df if 'AccelerometerStat' in column and column.endswith('DC_FFT')]].to_string()
        )'''

FEATURES = [
    'AccelerometerStat_x_FIRST_VAL_FFT',
    'AccelerometerStat_x_SECOND_VAL_FFT',
    'AccelerometerStat_x_THIRD_VAL_FFT',
    'AccelerometerStat_x_FOURTH_VAL_FFT',
    'AccelerometerStat_y_FIRST_VAL_FFT',
    'AccelerometerStat_y_SECOND_VAL_FFT',
    'AccelerometerStat_y_THIRD_VAL_FFT',
    'AccelerometerStat_y_FOURTH_VAL_FFT',
    'AccelerometerStat_z_FIRST_VAL_FFT',
    'AccelerometerStat_z_SECOND_VAL_FFT',
    'AccelerometerStat_z_THIRD_VAL_FFT',
    'AccelerometerStat_z_FOURTH_VAL_FFT']
''',
    'GyroscopeStat_x_FIRST_VAL_FFT',
    'GyroscopeStat_x_SECOND_VAL_FFT',
    'GyroscopeStat_x_THIRD_VAL_FFT',
    'GyroscopeStat_x_FOURTH_VAL_FFT',
    'GyroscopeStat_y_FIRST_VAL_FFT',
    'GyroscopeStat_y_SECOND_VAL_FFT',
    'GyroscopeStat_y_THIRD_VAL_FFT',
    'GyroscopeStat_y_FOURTH_VAL_FFT',
    'GyroscopeStat_z_FIRST_VAL_FFT',
    'GyroscopeStat_z_SECOND_VAL_FFT',
    'GyroscopeStat_z_THIRD_VAL_FFT',
    'GyroscopeStat_z_FOURTH_VAL_FFT',
    'MagneticField_x_FIRST_VAL_FFT',
    'MagneticField_x_SECOND_VAL_FFT',
    'MagneticField_x_THIRD_VAL_FFT',
    'MagneticField_x_FOURTH_VAL_FFT',
    'MagneticField_y_FIRST_VAL_FFT',
    'MagneticField_y_SECOND_VAL_FFT',
    'MagneticField_y_THIRD_VAL_FFT',
    'MagneticField_y_FOURTH_VAL_FFT',
    'MagneticField_z_FIRST_VAL_FFT',
    'MagneticField_z_SECOND_VAL_FFT',
    'MagneticField_z_THIRD_VAL_FFT',
    'MagneticField_z_FOURTH_VAL_FFT' '''
#]

df[FEATURES].to_csv('T2_features.csv', index=False)
