RECORD_MAP = {
    'Location' : {'frame' : {'record_path' : ['locations'], 'meta' : ['senseStartTimeMillis', 'userid', 'configAccuracy']},
                'columns_sel' : ['senseStartTimeMillis', 'userid', 'latitude', 'longitude', 'bearing', 'speed', 'altitude', 'provider','configAccuracy'],
                'target_table' : 'PHONE_LOCATIONS'},
    'PassiveLocation' : {'frame' : {'record_path' : None, 'meta' : None},
                'columns_sel' : ['senseStartTimeMillis', 'userid', 'latitude', 'longitude', 'bearing', 'speed', 'altitude', 'provider','accuracy'],
                'target_table' : 'PHONE_LOCATIONS'},                
    'InstalledApps' : {'frame' : {'record_path' : ['apps'], 'meta' : ['timestamp', 'userid']},
                    'columns_sel' : ['timestamp', 'userid', 'packageName', 'appName', 'isPreInstalled', 'versionName', 'firstInstallTime', 'lastUpdateTime','isInactive','isRunning', 'isPersistent'],
                    'target_table' : 'PHONE_APPLICATIONS'},
    'ActiveApps' : {'frame' : {'record_path' : ['apps'], 'meta' : ['timestamp', 'userid']},
                'columns_sel' : ['timestamp', 'userid', 'packageName', 'appName', 'isPreInstalled', 'versionName', 'firstInstallTime', 'lastUpdateTime','isInactive','isRunning', 'isPersistent'],
                'target_table' : 'PHONE_APPLICATIONS'},
    'AppUsageStats' : {'frame' : {'record_path' : None, 'meta' : None}, #{'record_path' : ['stats'], 'meta' : ['timestamp', 'userid']}
            'columns_sel' : ['timestamp', 'userid','stats'],
            'target_table' : 'APP_USAGE_STATS'},   
    'WiFi' : {'frame' : {'record_path' : ['scanResult'], 'meta' : ['senseStartTimeMillis', 'userid']},
            'columns_sel' : ['senseStartTimeMillis', 'userid', 'ssid', 'bssid', 'capabilities', 'frequency', 'level'],
            'target_table' : 'PHONE_WIFI_VISIBLE'},
    'Light' : {'frame' : {'record_path' : None, 'meta' : None},
                'columns_sel' : ['senseStartTimeMillis', 'userid', 'light', 'accuracy'],
                'target_table' : 'PHONE_LIGHT'},
    'PhysicalActivity' : {'frame' : {'record_path' : None, 'meta' : None},
                        'columns_sel' : ['senseStartTimeMillis', 'userid', 'activityName', 'activityType', 'confidence'],
                        'target_table' : 'PHONE_ACTIVITY_RECOGNITION'}, #special processing is done separately for this
    'Proximity' : {'frame' : {'record_path' : None, 'meta' : None},
            'columns_sel' : ['senseStartTimeMillis', 'userid', 'distance', 'maxRange'],
            'target_table' : 'PHONE_PROXIMITY'},
    'Screen' : {'frame' : {'record_path' : None, 'meta' : None},
            'columns_sel' : ['senseStartTimeMillis', 'userid', 'status'],
            'target_table' : 'PHONE_SCREEN'},          
    'Calendar' : {'frame' : {'record_path' : None, 'meta' : None}, #{'record_path' : ['events'], 'meta' : ['timestamp', 'userid']}
            'columns_sel' : ['timestamp', 'userid', 'events'],
            'target_table' : 'PHONE_CALENDAR'},     
    'PhoneRadio' : {'frame' : {'record_path' : None, 'meta' : None}, #{'record_path' : ['phoneRadioResult'], 'meta' : ['timestamp', 'userid']}
            'columns_sel' : ['senseStartTimeMillis', 'userid', 'phoneRadioResult'],
            'target_table' : 'PHONE_RADIO'},    
    'Battery' : {'frame' : {'record_path' : None, 'meta' : None},
            'columns_sel' : ['senseStartTimeMillis', 'userid','status','level','scale'],
            'target_table' : 'PHONE_BATTERY'},   
    'Bluetooth' : {'frame' : {'record_path' : ['devices'], 'meta' : ['userid']},
            'columns_sel' : ['timeStamp', 'userid','address','name','rssi'],
            'target_table' : 'PHONE_BLUETOOTH'}, 
    'Accelerometer' : {'frame' : {'record_path' : None, 'meta' : None},
            'columns_sel' : ['sensorTimeStamps', 'userid','xAxis','yAxis','zAxis'],
            'target_table' : 'PHONE_ACCELEROMETER'}

            
}     
