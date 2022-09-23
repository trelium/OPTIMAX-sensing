library(Amelia)
df <- read.csv('/home/jmocel/trelium/rapids/data/data/processed/features/all_participants/sel_sensor_features_nas_withtarget.csv')

df$progsegment <- ave(df$OASIS, df$pid, FUN = seq_along)


textvars <- c('local_segment', 'local_segment_label',  'local_segment_start_datetime', 'local_segment_end_datetime','segment_end_date', 'sensing_period', 'exp_group')


imputed_df <- amelia(df, m = 5, #generate 10 sets per participant 
                    ts = 'progsegment', 
                    noms = 'phone_activity_recognition_rapids_mostcommonactivity',
                    ncpus = 20, 
                    polytime = 3, 
                    intercs = TRUE, 
                    cs='pid',
                    idvars = textvars,
                    tolerance = 1e-03) 

write.amelia(obj=imputed_df, file.stem = "outdata")            

save(imputed_df, file = "imputations.RData")