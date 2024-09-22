import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data():
   df=pd.read_csv('../dataset/Collisions.csv',delimiter=',', index_col=False)
   # print(df)

   data = df.fillna('N/A')
   data = data.drop(['X','Y','OBJECTID','REPORTNO','EXCEPTRSNCODE','SDOTCOLNUM','INTKEY','LOCATION','INCKEY','COLDETKEY','SEGLANEKEY','CROSSWALKKEY'], axis=1)
   le = LabelEncoder()
   data['SEVERITYCODE'] = le.fit_transform(data['SEVERITYCODE'])
   data['ADDRTYPE'] = le.fit_transform(data['ADDRTYPE'])
   data['SEVERITYDESC'] = le.fit_transform(data['SEVERITYDESC'])
   data['JUNCTIONTYPE'] = le.fit_transform(data['JUNCTIONTYPE'])
   data['SDOT_COLDESC'] = le.fit_transform(data['SDOT_COLDESC'])
   data['INATTENTIONIND'] = le.fit_transform(data['INATTENTIONIND'])
   data['UNDERINFL'] = le.fit_transform(data['UNDERINFL'])
   data['WEATHER'] = le.fit_transform(data['WEATHER'])
   data['ROADCOND'] = le.fit_transform(data['ROADCOND'])
   data['LIGHTCOND'] = le.fit_transform(data['LIGHTCOND'])
   data['SPEEDING'] = le.fit_transform(data['SPEEDING'])
   data['ST_COLDESC'] = le.fit_transform(data['ST_COLDESC'])
   data['HITPARKEDCAR'] = le.fit_transform(data['HITPARKEDCAR'])
   data['COLLISIONTYPE'] = le.fit_transform(data['COLLISIONTYPE'])
   data['ST_COLCODE'] = le.fit_transform(data['ST_COLCODE'])
   data['PEDROWNOTGRNT'] = le.fit_transform(data['PEDROWNOTGRNT'])

   pd.set_option('future.no_silent_downcasting', True)

   data['SDOT_COLCODE'] = data['SDOT_COLCODE'].replace('N/A', 0)
   data['SDOT_COLCODE'] = data['SDOT_COLCODE'].astype(int)
   data['SDOT_COLCODE'] = le.fit_transform(data['SDOT_COLCODE'])

   data_corr = data[['SEVERITYCODE','ADDRTYPE', 'COLLISIONTYPE', 'PERSONCOUNT',
      'PEDCOUNT', 'PEDCYLCOUNT', 'VEHCOUNT', 'INJURIES', 'SERIOUSINJURIES',
      'FATALITIES','JUNCTIONTYPE', 'SDOT_COLCODE',
      'INATTENTIONIND', 'UNDERINFL', 'WEATHER', 'ROADCOND',
      'LIGHTCOND', 'PEDROWNOTGRNT', 'SPEEDING', 'ST_COLCODE', 'HITPARKEDCAR']]
   # print(data_corr)
   return df,data_corr