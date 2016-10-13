# Map 1-based optional input ports to variables
dataset1 = maml.mapInputPort(1) # class: data.frame
test = maml.mapInputPort(2) # class: data.frame

require(xgboost)
require(Matrix)
require(data.table)

# do the same data preparation for the testing data as what I did for the training data
test[is.na(test)] = -9999
test.dt=data.table(test)

label_mapping=data.table(unique_label=c(111, 121, 122 ,211 ,212 ,221 ,222, 231, 241 ,311, 312, 321 ,322, 411, 412, 511 ,512 ,522, 531, 532, 611, 612 ,621, 711,712, 721, 722, 731, 811 ,821 ,831, 841, 911, 912,921, 931 ,932),
  order=c(26 ,19 , 9, 28, 36 , 3 ,16 , 2 ,25,  4 ,31, 34, 20 , 1 ,21 ,18 ,17 ,10, 13, 33  ,7 ,12, 11, 35, 22 ,32, 14 ,23 ,15 , 6 , 5 ,29 , 8,27 ,24 , 0, 30))
setkey(label_mapping,'unique_label')

rel_map=data.table(rel=sort(unique(test.dt$religion)),rel_num=c(1:length(unique(test.dt$religion))))
if(nrow(rel_map)!=11) {stop}
setkey(rel_map,'rel')
setkey(test.dt,'religion')

test.dt=test.dt[rel_map]
test.dt[,religion:=rel_num]
test.dt$rel_num=NULL

test.df=data.frame(test.dt)

# select the features
feature_index= c(2:18, 20:(50-2))
summary(test.df[,feature_index])
dtest=xgb.DMatrix(data=data.matrix(test.df[,feature_index])+0.0,missing=-9999)
print('data prepared')

# a function to map the predicted labels to the geo,segment and subgroup label
class_to_label=function(class,label_mapping){
    lm=label_mapping$unique_label
    names(lm)=label_mapping$order
    y=unname(lm[as.character(class)])
    geo=as.integer(substring(y,1,1))
    segment=as.integer(substring(y,2,2))
    subgroup=as.integer(substring(y,3,3))
    return(list(geo=geo,segment=segment,subgroup=subgroup))
}

# load the saved xgboost model
raw_model = as.raw(unlist(dataset1[1]))
model_xgb = xgb.load(raw_model) 

# predict 
pred_xgb=predict(model_xgb,dtest)
# map labels
label_xgb=class_to_label(pred_xgb,label_mapping)
# create the output data.frame
label_xgb_df=data.frame('patientID'=test.dt$patientID,'Geo_Pred'=label_xgb$geo,'Segment_Pred'=label_xgb$segment,'Subgroup_Pred'=label_xgb$subgroup)


# Select data.frame to be sent to the output Dataset port
maml.mapOutputPort("label_xgb_df")
