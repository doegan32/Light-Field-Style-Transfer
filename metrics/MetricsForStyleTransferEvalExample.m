%% Open light field
% Table
dataPathOrg = 'path/to/original/lightfield/'; imgNameOrg = 'input_Cam';
dataPathST = 'path/to/stylized/lightfield/'; imgNameST = 'input_Cam';
dataPathDisp = 'path/to/disparity/maps/'; NameDisp = 'disparity.pfm'

height = 512;
width = 512;
rows = 9;
cols = 9;
sc = ceil(cols/2);
tc = ceil(rows/2);
LFOrg = zeros(rows,cols,height,width,3);
LFST  = zeros(rows,cols,height,width,3);
DF = zeros(rows,cols,height,width);

st_idx = 0;
tic
for t = 1:rows
    for s = 1:cols
        % Original light field
        %imName = [dataPathOrg sprintf('%s_%02d_%02d.png', imgNameOrg, t-1, s-1)];
        imName = [dataPathOrg sprintf('%s%03d.png', img_name, st_idx)];
        disp(['Reading ' imName])
        LFOrg(t,s,:,:,:) = double(imread(imName));
                
        % Stylized light field
        % imName = [dataPathST sprintf('%s_%02d_%02d.png', img_name, t-1, s-1)];
        imName = [dataPathST sprintf('%s%03d.png', img_name, st_idx)];
        disp(['Reading ' imName])
        LFST(t,s,:,:,:) = double(imread(imName));
        
        % Disparity map
        dispName = [dataPathDisp sprintf('DISP_MAP_%03d.pfm', st_idx)];
        disp(['Reading ' dispName])
        DF(t,s,:,:) = pfmread(dispName);
        
        st_idx = st_idx+1;
    end
end

dispMap = squeeze(DF(sc,tc,:,:));

% dispMapPath = fullfile(dataPathDisp, NameDisp);
% disp(['Reading ' dispMapPath])
% dispMap = pfmread(dispMapPath);
toc

%% Light Field Epipolar Consistency metric
tic
[LFECOrg, LFOrgVar2, LFOrgWarp, LFOrgWarpmask] = LightFieldEpipolarConsistency(LFOrg, dispMap, 'linear');
[LFECST, LFSTVar2, LFSTWarp, LFSTWarpmask] = LightFieldEpipolarConsistency(LFST, dispMap, 'linear');
toc

LFOrgVar2 = mean(LFOrgVar2,3);
LFSTVar2 = mean(LFSTVar2,3);
LFVar2range = [min([LFOrgVar2(:); LFSTVar2(:)]) max([LFOrgVar2(:); LFSTVar2(:)])];

figure
subplot(1,2,1), imshow(LFOrgVar2, LFVar2range), colormap(jet), colorbar, title(LFECOrg)
subplot(1,2,2), imshow(LFSTVar2, LFVar2range), colormap(jet), colorbar, title(LFECST)


%% Light Field Angular Consistency metric (LOCAL)
tic
[LFACOrg, LFOrgVar2] = LightFieldAngularConsistency(LFOrg, DF, false, 'linear');
[LFACST, LFSTVar2] = LightFieldAngularConsistency(LFST, DF, false, 'linear');
toc

LFOrgVar2 = mean(LFOrgVar2,3);
LFSTVar2 = mean(LFSTVar2,3);
LFVar2range = [min([LFOrgVar2(:); LFSTVar2(:)]) max([LFOrgVar2(:); LFSTVar2(:)])];

figure
subplot(1,2,1), imshow(LFOrgVar2, LFVar2range), colormap(jet), colorbar, title(LFACOrg)
subplot(1,2,2), imshow(LFSTVar2, LFVar2range), colormap(jet), colorbar, title(LFACST)
