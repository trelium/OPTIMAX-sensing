RECORD_MAP = {
    'Location' : {'frame' : {'record_path' : ['locations'], 'meta' : ['senseStartTimeMillis', 'userid', 'configAccuracy']},
                'columns_sel' : ['senseStartTimeMillis', 'userid', 'latitude', 'longitude', 'bearing', 'speed', 'altitude', 'provider','configAccuracy']},
    'InstalledApps' : {'frame' : {'record_path' : ['apps'], 'meta' : ['timestamp', 'userid']},
                    'columns_sel' : ['timestamp', 'userid', 'packageName', 'appName', 'isPreInstalled', 'versionName', 'firstInstallTime', 'lastUpdateTime','isInactive','isRunning', 'isPersistent']},
    'ActiveApps' : {'frame' : {'record_path' : ['apps'], 'meta' : ['timestamp', 'userid']},
                'columns_sel' : ['timestamp', 'userid', 'packageName', 'appName', 'isPreInstalled', 'versionName', 'firstInstallTime', 'lastUpdateTime','isInactive','isRunning', 'isPersistent']},
    'WiFi' : {'frame' : {'record_path' : ['scanResult'], 'meta' : ['senseStartTimeMillis', 'userid']},
            'columns_sel' : ['senseStartTimeMillis', 'userid', 'ssid', 'bssid', 'capabilities', 'versionName', 'frequency', 'level']},
    'Light' : {'frame' : {'record_path' : None, 'meta' : None},
                'columns_sel' : ['senseStartTimeMillis', 'userid', 'light', 'accuracy']},
    'PhysicalActivity' : {'frame' : {'record_path' : None, 'meta' : None},
                            'columns_sel' : ['senseStartTimeMillis', 'userid', 'activityName', 'activityType', 'confidence']}, #special processing is done separately for this
    'Proximity' : {'frame' : {'record_path' : None, 'meta' : None},
            'columns_sel' : ['senseStartTimeMillis', 'userid', 'distance', 'maxRange']},
    'Screen' : {'frame' : {'record_path' : None, 'meta' : None},
            'columns_sel' : ['senseStartTimeMillis', 'userid', 'status']},          
    'Calendar' : {'frame' : {'record_path' : ['events'], 'meta' : ['timestamp', 'userid']},
            'columns_sel' : ['timestamp', 'userid']},     
    'PhoneRadio' : {'frame' : {'record_path' : ['phoneRadioResult'], 'meta' : ['senseStartTimeMillis', 'userid']},
            'columns_sel' : ['senseStartTimeMillis', 'userid']},    
    'AppUsageStats' : {'frame' : {'record_path' : ['stats'], 'meta' : ['timestamp', 'userid']},
            'columns_sel' : ['timestamp', 'userid']},   
    'Battery' : {'frame' : {'record_path' : None, 'meta' : None},
            'columns_sel' : ['senseStartTimeMillis', 'userid','status','level','scale']},   
    'Bluetooth' : {'frame' : {'record_path' : ['devices'], 'meta' : ['userid']},
            'columns_sel' : ['timeStamp', 'userid','address','name','rssi']}, 
    'Accelerometer' : {'frame' : {'record_path' : None, 'meta' : None},
            'columns_sel' : ['sensorTimeStamps', 'userid','xAxis','yAxis','zAxis']}
            
            
}     
