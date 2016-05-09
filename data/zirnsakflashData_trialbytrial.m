function [] = zirnsakflashData_trialbytrial (driveLetter,expNom,repeat,chan)

%Parameters to check:
% align responses to saccade onset or offset?
% time window tolerance to discard saccade-P2probe overlap


disp(['chan ' num2str(chan) ' is gooowaaddd'])
plottt=1;
getlfp=0;
fs=500;
binwidth=.025;
timewin=.35;
edges=-timewin:binwidth:timewin;
edgesl=linspace(-timewin*fs,timewin*fs,length(-timewin*fs:timewin*fs));
%  baseidx=4:7;%for 50ms bins;vis_lat=11;
baseidx=4:12;%for 25ms bins
vis_lat=18;
% baseidx=9:17;%for 25ms bins, -.45 to .45
% vis_lat=26;
for r=1:length(repeat)
    expName = strcat(expNom,repeat{r});
    spikeData = sprintf('~/ArrayData/%s/%s/%s/mat/sorted_%s.mat',driveLetter,expName(1:4),expName,expName);
    load(spikeData,'cluster_class');
    plxEvts = sprintf('~/ArrayData/%s/%s/%s/mat/%s.mat',driveLetter,expName(1:4),expName,expName);
    load(plxEvts,'eventtimes')
    eyeSig = sprintf('/home/packremote/SharedDocuments/Sujay/experiments/%s/%s_processedEye1.mat',expName(1:4),expName(end));
    load (eyeSig)
    mlEvts = sprintf('~/ArrayData/%s/%s/%s/ml/processedEvents.mat',driveLetter,expName(1:4),expName);
    load(mlEvts,'tgain','toffset');
    mlEvts = sprintf('~/ArrayData/%s/%s/%s/ml/grpEvents.mat',driveLetter,expName(1:4),expName);
    load(mlEvts,'makeStim','startCollectingData','fixflashed','fix2flashed','sac1presented','sac1probepresented');
    
    %sync times
    plxtoml = @(x) tgain*x + toffset;
    mltoplx = @(x) (x - toffset)/tgain;
    t0=startCollectingData.time;
    
    
    spkinf = cluster_class{chan};
    if isempty(spkinf), continue;end
    spktimes = spkinf(:,2); %could check for multiple spikes on the same electrode here
    
    if getlfp
        if chan>9,fname=sprintf('~/ArrayData/%s/%s/%s/lfps/%s_ch0%d',driveLetter,expName(1:4),expName,'lfp',chan);
        else fname=sprintf('~/ArrayData/%s/%s/%s/lfps/%s_ch00%d',driveLetter,expName(1:4),expName,'lfp',chan);
        end
        try
            fid=fopen(fname);
            wb=fread(fid,Inf,'double');
            fclose(fid);
        catch
            disp (['LFP of chan ' num2str(chan) ' is not goood']) ;
            return;
        end
        
    end
    spktimes= plxtoml(spktimes/1000)+0.02;
    idx=find(success(:,1)~=0);
    if isempty(saclatency)
        saclatency = (makeStim.stprobe+makeStim.probedur+20)/1000*ones(size(success,1),2);
    end
    saclat=saclatency(idx,:);
    
    %P1SP2_times = ([P1_time(:)        P1_where(:)   logical_P1P2time(<mean of ST)(:)
    %sac_onset_time(:)  P2_time(:)    P2_where(:)
    %P3_time(:)         P3_where(:) ]);
    % how sac latency is stored:
    % lat=-[ref-sacONtime ref-sacOFFtime];
    % saclatency(i,:)=lat;
    for i = 1:(length(idx))
        ticksSac = sac1presented(idx(i)).flipTime-t0+saclat(i,2);%aligned to sacc offset
        ticksSacp2 = sac1probepresented(idx(i)).flipTime-t0;
        if saclat(i,1)<(makeStim.stprobe+makeStim.probedur+10)/1000 %tolerance of 10ms
            psthsac(i,:)=nan(length(edges)-1,1);
            lfpsac(i,:)=nan(length(edgesl),1);
            psthp2(i,:)=nan(length(edges)-1,1);
            lfpp2(i,:)=nan(length(edgesl),1);
            
            
        else
            temp = spktimes(spktimes>ticksSac-timewin & spktimes<ticksSac+timewin);
            temp=histc(temp-ticksSac,edges);
            psthsac(i,:)=temp(1:end-1)/binwidth;
            tempp2 = spktimes(spktimes>ticksSacp2-timewin & spktimes<ticksSacp2+timewin);
            tempp2=histc(tempp2-ticksSacp2,edges);
            psthp2(i,:)=tempp2(1:end-1)/binwidth;
            if getlfp
                tickplx = round(mltoplx(ticksSac)*fs);
                lfpsac(i,:)=wb(tickplx-timewin*fs:tickplx+timewin*fs);
                tickplx = round(mltoplx(ticksSacp2)*fs);
                lfpp2(i,:)=wb(tickplx-timewin*fs:tickplx+timewin*fs);
            end
            
        end
        
        ticksSac = P1SP3_times(i,1);
        temp = spktimes(spktimes>ticksSac-timewin  & spktimes<ticksSac+timewin);
        temp=histc(temp-ticksSac,edges);
        psth1(i,:)=temp(1:end-1)/binwidth;
        if getlfp
            tickplx = round(mltoplx(ticksSac)*fs);
            lfp1(i,:)=wb(tickplx-timewin*fs:tickplx+timewin*fs);
        end
    end
    idx=find(success(:,3)~=0);
    for i=1:length(idx)
        
        ticksSac = P3P(i,1);
        temp = spktimes(spktimes>ticksSac-timewin & spktimes<ticksSac+timewin);
        temp=histc(temp-ticksSac,edges);
        psth3(i,:)=temp(1:end-1)/binwidth;
        if getlfp
            tickplx = round(mltoplx(ticksSac)*fs);
            lfp3(i,:)=wb(tickplx-timewin*fs:tickplx+timewin*fs);
        end
        
    end
    p1p2_times{r}=P1SP3_times;
    p3_times{r}=P3P;
    
    PSTHsac{r}= psthsac;
    PSTH2{r}= psthp2;
    PSTH1{r}= psth1;
    PSTH3{r}= psth3;
     
    if getlfp
        LFPsac{r}= lfpsac;
        LFP2{r}= lfpp2 ;
        LFP1{r}= lfp1 ;
        LFP3{r}= lfp3 ;
    end
    clear P1SP3_times psthsac psth1 psth3 lfpsac lfp1 lfp3
end

edges=edges(1:end-1);
if ~isfield(makeStim,'probeSpacing'), makeStim.probeSpacing=4;end

rect=makeStim.probeRect;
stimx = rect(1)+makeStim.probeSpacing:makeStim.probeSpacing:rect(1)+rect(3);
stimy = rect(2)-makeStim.probeSpacing:-makeStim.probeSpacing:rect(2)-rect(4);
if ~isfield(makeStim,'ort'), makeStim.ort=0;end
if ~isfield(makeStim,'sf'), makeStim.sf=0;end
if makeStim.sf==0, makeStim.ort=0;end
numProbes = length(stimx)*length(stimy)*length(makeStim.ort)+1;
if makeStim.fixX <=0
    saccadeleft=0;
else
    saccadeleft=1;
end

numCols=length(stimx);
psth1=cell(numProbes,1);psth2=psth1;psth3=psth1;psthsac=psth1;
lfp1=psth1;lfp2=psth1;lfp3=psth1;lfpsac=lfp1;

for r=1:length(repeat)
    tempsac=PSTHsac{r};
    temp2=PSTH2{r};
    temp1=PSTH1{r};
    temp3=PSTH3{r};
    if getlfp
        templfp2=LFP2{r};
        templfpsac=LFPsac{r};
        templfp1=LFP1{r};
        templfp3=LFP3{r};
    end
    for loc=1:numProbes
        
        p2loc = p1p2_times{r}(:,6)==loc;
        p1loc = p1p2_times{r}(:,2)==loc;
        %p3loc = p1p2_times{r}(:,8)==loc; % if using conservative method
        %of taking only perfect trials with successful P1, P2 and P3
        p3loc = p3_times{r}(:,2)==loc;
        
        tempp=temp1(p1loc,:);
        psth1{loc}= cat(1,psth1{loc},tempp);
        tempp=tempsac(p2loc,:);
        psthsac{loc}= cat(1,psthsac{loc},tempp);
        tempp=temp2(p2loc,:);
        psth2{loc}= cat(1,psth2{loc},tempp);
        tempp=temp3(p3loc,:);
        psth3{loc}= cat(1,psth3{loc},tempp);
        if makeStim.fixY==makeStim.fixY2 && makeStim.fixX==makeStim.fixX2
            psth1{loc}= cat(1,psth1{loc},temp1(p1loc,:),temp2(p2loc,:),temp3(p3loc,:));
        end
        
        if getlfp
            tempp=templfp1(p1loc,:);
            lfp1{loc}= cat(1,lfp1{loc},tempp);
            tempp=templfpsac(p2loc,:);
            lfpsac{loc}= cat(1,lfpsac{loc},tempp);
            tempp=templfp2(p2loc,:);
            lfp2{loc}= cat(1,lfp2{loc},tempp);
            tempp=templfp3(p3loc,:);
            lfp3{loc}= cat(1,lfp3{loc},tempp);
        end
    end
    
    
end


cd /export/SharedDocuments/Yann/data
try
    load([expName(1:4) '_' strcat(repeat{:}) '.mat']);
catch
    disp 'making a new file'
    fr1=cell(numProbes,1);fr2=fr1;fr3=fr1;frsac=fr1;
end
for loc=1:numProbes
    [a,~]=find(~isnan(psth2{loc}(:,1)));
    
    fr1{loc}(chan,:,:)=psth1{loc};
    fr2{loc}(chan,:,:)=psth2{loc}(a,:);
    fr3{loc}(chan,:,:)=psth3{loc};
    frsac{loc}(chan,:,:)=psthsac{loc}(a,:);
    
end
save([expName(1:4) '_' strcat(repeat{:}) '.mat'],'fr1','fr2','fr3','frsac',...
    'edges','stimx','stimy','makeStim');


end


% data structure: {channel x trial x time} x 101 locations
% probelocations:  stimx and stimy
% fixation points: makeStim.fixX, makeStim.fixX2 etc. 
% other expt parameter: in makeStim structure 