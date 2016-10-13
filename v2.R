#########################################################
########## load the train data and train model ##########
#########################################################
colclasses <- rep("integer",50)
colclasses[36] <- "character"
train <- read.table('WomenHealth_Training.csv', header=TRUE, sep=",", strip.white=TRUE, stringsAsFactors = F, colClasses = colclasses)

require(xgboost)
require(data.table)
require(Matrix)
require(CatEncoders)

train[is.na(train)] = -9999

# create a data.table format for the train data
train.dt=data.table(train)

# create the unique labels from geo, segment and subgroup
unique_label = train.dt[,unique(100*geo+10*segment+subgroup)]
# number of unique classes
mclass=length(unique_label)

# since the predicted classes from xgb is 0..(mclass-1), mapping between unique_label and predicted classes is needed.

# create y
lenc1=LabelEncoder.fit(train.dt[,100*geo+10*segment+subgroup])
train_y=transform(lenc1,train.dt[,100*geo+10*segment+subgroup])-1

# how many features in train data
ncols = ncol(train)

# select all the available features
feature_index= c(2:18, 20:(ncols-2))

########## deal with character feature religion by converting the religion column to integer #########
lenc2=LabelEncoder.fit(train.dt[,religion])

train.dt[,religion:=transform(lenc2,religion)]

# train data ready to use
print('data prepared')

# convert the data from data.table to data.frame 
train.df=data.frame(train.dt)

summary(train.df[,feature_index])

set.seed(2016)
dtrain=xgb.DMatrix(data=data.matrix(train.df[,feature_index])+0.0,label=train_y,missing=-9999)

print('start xgb')

# set the parameters for xgboost
params=list(
  booster='gbtree',
  objective='multi:softmax',
  lambda=15,
  subsample=0.8,
  colsample_bytree=0.75,
  min_child_weight=2,
  max_depth=8,
  eta=0.06,
  eval_metric='merror',
  num_class=37
)
# number of rounds to use
nrounds=360
model_xgb=xgb.train(params = params,data=dtrain,nrounds = nrounds)


#########################################################
############ load the test data and predict #############
#########################################################
test = train # class: data.frame


# do the same data preparation for the testing data as what I did for the training data
test[is.na(test)] = -9999
test.dt=data.table(test)

test.dt[,religion:=transform(lenc2,religion)]

test.df=data.frame(test.dt)

# select the features
feature_index= c(2:18, 20:(50-2))
summary(test.df[,feature_index])
dtest=xgb.DMatrix(data=data.matrix(test.df[,feature_index])+0.0,missing=-9999)
print('data prepared')

# a function to map the predicted labels to the geo,segment and subgroup label
class_to_label=function(class,lenc){
  y=as.character(inverse.transform(lenc,pred_xgb+1))
  geo=as.integer(substring(y,1,1))
  segment=as.integer(substring(y,2,2))
  subgroup=as.integer(substring(y,3,3))
  return(list(geo=geo,segment=segment,subgroup=subgroup))
}

# predict 
pred_xgb=predict(model_xgb,dtest)

# map labels
label_xgb=class_to_label(pred_xgb,lenc1)
# create the output data.frame
label_xgb_df=data.frame('patientID'=test.dt$patientID,'Geo_Pred'=label_xgb$geo,'Segment_Pred'=label_xgb$segment,'Subgroup_Pred'=label_xgb$subgroup)
