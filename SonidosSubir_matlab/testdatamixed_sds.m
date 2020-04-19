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
    %for j=1:length(sub_d)-2
        
        %sub_sub_d=struct;
        %sub_sub_d=dir([data_dir,directory(class+2).name,'/',sub_d(j+2).name]);
    %end
        %for i=1:length(sub_sub_d)-2
     %for j=1:length(sub_d)-2       
            %tmp1=struct2cell(sub_sub_d);
%             tmp=struct2cell(sub_d);
%             %tmp1=tmp1(1,:)';
%             tmp=tmp(1,:)';
%             myindices = ~cellfun(@isempty,regexp(tmp,'S3'));
%             sub_train=tmp(~myindices);
%             sub_test=tmp(myindices);
           %nfile=length(sub_train)-2; 
        %end
    %end
%end   
%% Training     
    for file=1:4 
        %for i=1:length(sub_sub_d)-2
        
        %if((sub_d(file+2).name) ~= 'S3' )
            sub_sub_d=dir([data_dir,directory(class+2).name,'\',sub_d(file+2).name]);
        
            for i=1:length(sub_sub_d)-2
            
                 [x,fs]=audioread([data_dir,directory(class+2).name,'\',sub_d(file+2).name,'\',sub_sub_d(i+2).name]);
%                 %Z1=feature_extraction(x,fs);
                Z11=graphsig_tmp(x,fs);
                 A=imresize(Z11,[100,100]);
                 B=reshape(A,[1,10000]);
%         I=imread([data_dir,directory(class+2).name,'/',sub_d(file+2).name]);
%                [x,fs]=audioread([data_dir,directory(class+2).name,'\',sub_train{file+2},'\',sub_sub_d(i+2).name]);
                Z1=gfcc_extraction(x,fs);
%         I=imread([data_dir,directory(class+2).name,'/',sub_d(file+2).name]);
%       
%       
                temp1=Z1.stat.mean;
                %temp11=reshape(temp1,[1,39]);
                temp2=Z1.stat.std;
                %temp22=reshape(temp2,[1,39]);
                V=[temp1' temp2' B];
                train=[train;V];
                LabelsTrain=[LabelsTrain;class]   
%    
%                 temp1=Z1.stat.mean;
%                 temp2=Z1.stat.std;
%                 V=[temp1' temp2'];
%                 train=[train;B];
%                 LabelsTrain=[LabelsTrain;class];
            end
        %end
        
%        catch exception
%           failedfiles=[failedfiles;{file}];
%         continue
%       end
    fprintf('Done reading %s class training files\n',sub_d(file+2).name);    
    end;
end

%% Testing 


data_dir1='C:\Users\ARKADIP GHOSH\Desktop\matlab project\sonidosSubir\allSounds\testdatamixed\wavs-mix5\'; %this is where the clean sounds are stored
directory1=dir(data_dir1);
nclass1=length(directory1)-2;
failedfiles={};
%nfile=50

%train=[];LabelsTrain=[];test=[];LabelsTest=[];
t = cputime;

for class=1:nclass1
    sub_d1=dir([data_dir1,directory1(class+2).name]);
    for i=1:length(sub_d1)-2
            
                [x,fs]=audioread([data_dir1,directory1(class+2).name,'\',sub_d1(i+2).name]);
                Z2=gfcc_extraction(x,fs);
                 Z22=graphsig_tmp(x,fs);
                 A1=imresize(Z22,[100,100]);
                 B1=reshape(A1,[1,10000]);
                temp3=Z2.stat.mean;
                %temp33=reshape(temp3,[1,39]);
                 temp4=Z2.stat.std;
                 %temp44=reshape(temp4,[1,39]);
                 V1=[temp3' temp4' B1];
                test=[test;V1];
                LabelsTest=[LabelsTest;class];
            end
end
        
        
% %fprintf('Done reading %s class testing files\n',sub_test{file});    
% %end

%load(['/home/manjunath/graph-features/water_tap/',num2str(1)])
fprintf('Done reading %d class training files\n',cputime-t);

%% Classification
%Ftrain=train;Ftest=test;
%traintest;
%mctrain;
mctrain
%chi_int;