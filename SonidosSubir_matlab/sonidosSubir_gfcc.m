clear all
clear
data_dir='C:\Users\ARKADIP GHOSH\Desktop\matlab project\sonidosSubir\allSounds\testdataclean\'; %this is where the clean sounds are stored
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
    for j=1:length(sub_d)-2
        
        %sub_sub_d=struct;
        sub_sub_d=dir([data_dir,directory(class+2).name,'/',sub_d(j+2).name]);
    %end
        %for i=1:length(sub_sub_d)-2
     %for j=1:length(sub_d)-2       
            %tmp1=struct2cell(sub_sub_d);
            tmp=struct2cell(sub_d);
            %tmp1=tmp1(1,:)';
            tmp=tmp(1,:)';
            myindices = ~cellfun(@isempty,regexp(tmp,'S4'));
            sub_train=tmp(~myindices);
            sub_test=tmp(myindices);
            nfile=length(sub_train)-2; 
        %end
    end
%end   
%% Training     
    for file=1:nfile
        %for i=1:length(sub_sub_d)-2
        sub_sub_d=dir([data_dir,directory(class+2).name,'\',sub_train{file+2}]);
        for i=1:length(sub_sub_d)-2
            
            [x,fs]=audioread([data_dir,directory(class+2).name,'\',sub_train{file+2},'\',sub_sub_d(i+2).name]);
            %Z1=feature_extraction(x,fs);
            Z1=gfcc_extraction(x,fs);
%         I=imread([data_dir,directory(class+2).name,'/',sub_d(file+2).name]);
            Z2=feature_extraction(x,fs);
%       
            temp1=Z1.stat.mean;
            temp2=Z1.stat.std;
            temp3=Z2.stat.mean;
            temp4=Z2.stat.std;
            V=[temp1' temp2' temp3' temp4'];
            train=[train;V];
            LabelsTrain=[LabelsTrain;class];
        end
     
        
%        catch exception
%           failedfiles=[failedfiles;{file}];
%         continue
%       end
    fprintf('Done reading %s class training files\n',sub_train{file+2});    
    end;


%% Testing 
%clear nfile V Z1 x fs temp1 temp2
% nfile1=length(sub_test);
%for file1=1:nfile1
    sub_sub_d1=dir([data_dir,directory(class+2).name,'\','S4']);
    for i=1:length(sub_sub_d1)-2
       %try
           [x,fs]=audioread([data_dir,directory(class+2).name,'\',sub_test{1},'\',sub_sub_d1(i+2).name]);
            %Z2=feature_extraction(x,fs);
            Z3=gfcc_extraction(x,fs);
%         I=imread([data_dir,directory(class+2).name,'/',sub_d(file+2).name]);
            Z4=feature_extraction(x,fs);
%       
        temp5=Z3.stat.mean;
        temp6=Z3.stat.std;
        temp7=Z4.stat.mean;
        temp8=Z4.stat.std;
        V=[temp5' temp6' temp7' temp8'];
         test=[test;V];
        LabelsTest=[LabelsTest;class];
%        catch exception
%           failedfiles=[failedfiles;{file}];
%         continue
    end
%fprintf('Done reading %s class testing files\n',sub_test{file});    
%end
end
%load(['/home/manjunath/graph-features/water_tap/',num2str(1)])
fprintf('Done reading %d class training files\n',cputime-t);

%% Classification
%Ftrain=train;Ftest=test;
%traintest;
mctrain;
%mctrain
%chi_int;