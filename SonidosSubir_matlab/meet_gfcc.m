clear all
clear
data_dir='/media/manjunath/BCE0E709E0E6C8AA/Evaluation-data/meeting-folds/'; %this is where the clean sounds are stored
directory=dir(data_dir);
nclass=length(directory)-2;
failedfiles={};
%nfile=50
fold_num='1';
train=[];LabelsTrain=[];test=[];LabelsTest=[];
t = cputime;
%% Training Features
for class=1:nclass

    sub_d=dir([data_dir,directory(class+2).name]);
    tmp1=struct2cell(sub_d);
    tmp1=tmp1(1,:)';
    myindices = ~cellfun(@isempty,regexp(tmp1,'[0-9]+[-][5]\.wav'));
    sub_train=tmp1(~myindices);
    sub_test=tmp1(myindices);
    nfile=length(sub_train)-2; 
   
    
%% Training     
    for file=1:nfile
       %try
           [x,fs]=audioread([data_dir,directory(class+2).name,'/',sub_train{file+2}]);
             Z1=gfcc_extraction(x,fs);
%         I=imread([data_dir,directory(class+2).name,'/',sub_d(file+2).name]);
%       
%       
        temp1=Z1.stat.mean;
        temp2=Z1.stat.std;
        V=[temp1' temp2'];
         train=[train;V];
        LabelsTrain=[LabelsTrain;class];
%        catch exception
%           failedfiles=[failedfiles;{file}];
%         continue
%       end
fprintf('Done reading %s class training files\n',sub_train{file});    
    end;


%% Testing 
clear nfile V Z1 x fs temp1 temp2
 nfile=length(sub_test);
for file=1:nfile
       %try
           [x,fs]=audioread([data_dir,directory(class+2).name,'/',sub_test{file}]);
                    Z1=gfcc_extraction(x,fs);
%         I=imread([data_dir,directory(class+2).name,'/',sub_d(file+2).name]);
%       
%       
        temp1=Z1.stat.mean;
        temp2=Z1.stat.std;
        V=[temp1' temp2'];
         test=[test;V];
        LabelsTest=[LabelsTest;class];
%        catch exception
%           failedfiles=[failedfiles;{file}];
%         continue
%       end
fprintf('Done reading %s class testing files\n',sub_test{file});    
end;
end;
%load(['/home/manjunath/graph-features/water_tap/',num2str(1)])
fprintf('Done reading %d class training files\n',cputime-t);

%% Classification
mctrain
%chi_int;