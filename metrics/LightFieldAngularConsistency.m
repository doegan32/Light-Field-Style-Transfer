function [LFAC, LFWarpVar2] = LightFieldAngularConsistency(LF, DF, useLFEC, interp)

if nargin < 3
    useLFEC = true;
end

if nargin < 4
    interp = 'linear';
end

rows = size(LF,1);
cols = size(LF,2);
height = size(LF,3);
width = size(LF,4);
nch = size(LF,5);

[X, Y] = meshgrid(1:width, 1:height);

LFWarpVar2 = zeros(height, width, 3);
WarpMaskSum = zeros(height, width, 3);

% Locally Back-warping to center view from center disparity map
for t = 2:(rows-1)
    for s = 2:(cols-1)
        % Local variance computation
        LocalWarp = zeros(3,3,height,width,nch);
        LocalWarp(2,2,:,:,:) = squeeze(LF(t,s,:,:,:));
        LocalWarpmask = zeros(3,3,height,width,nch);
        LocalWarpmask(2,2,:,:,:) = ones(height, width, 3);
        dispMap = squeeze(DF(t,s,:,:));
        for tt = -1:1
            for ss = -1:1
                if tt == 0 && ss == 0, continue, end
                Xwarp = X - ss * dispMap;
                Ywarp = Y - tt * dispMap;

                for ch = 1:nch
                    LocalWarp(tt+2,ss+2,:,:,ch) = interp2(squeeze(LF(t+tt,s+ss,:,:,ch)), Xwarp, Ywarp, interp);
                end
                stmask = double(Xwarp<=width & Xwarp>=1 & Ywarp<=height & Ywarp>=1);
                LocalWarpmask(tt+2,ss+2,:,:,:) = cat(3,stmask,stmask,stmask);
            end
        end
        LocalWarpMean = squeeze(sum(sum(LocalWarp .* LocalWarpmask, 1), 2) ./ sum(sum(LocalWarpmask, 1), 2));
        
        % Add all local squared variances together
        for tt = 1:3
            for ss = 1:3
                LFWarpVar2 = LFWarpVar2 + (squeeze(LocalWarp(tt,ss,:,:,:)) - LocalWarpMean).^2 .* squeeze(LocalWarpmask(tt,ss,:,:,:));
                WarpMaskSum = WarpMaskSum + squeeze(sum(sum(LocalWarpmask, 1), 2));
            end
        end
    end
end

if useLFEC
    % Back-warping to center view from center disparity map
    sc = ceil(cols/2);
    tc = ceil(rows/2);
    LFWarp = zeros(size(LF));
    LFWarp(tc,sc,:,:,:) = squeeze(LF(tc,sc,:,:,:));
    LFWarpmask = zeros(size(LF));
    LFWarpmask(tc,sc,:,:,:) = ones(height, width, 3);
    
    for t = 1:rows
        for s = 1:cols
            if t == tc && s == sc, continue, end
            ds = sc - s;
            dt = tc - t;
            Xwarp = X + dispMap * ds;
            Ywarp = Y + dispMap * dt;
            
            for ch = 1:nch
                LFWarp(t,s,:,:,ch) = interp2(squeeze(LF(t,s,:,:,ch)), Xwarp, Ywarp, interp);
            end
            stmask = double(Xwarp<=width & Xwarp>=1 & Ywarp<=height & Ywarp>=1);
            LFWarpmask(t,s,:,:,:) = cat(3,stmask,stmask,stmask);
        end
    end
    % Compute back warp views variance map
    LFWarpMean = squeeze(sum(sum(LFWarp .* LFWarpmask, 1), 2) ./ sum(sum(LFWarpmask, 1), 2));
    for t = 1:rows
        for s = 1:cols
            LFWarpVar2 = LFWarpVar2 + (squeeze(LFWarp(t,s,:,:,:)) - LFWarpMean).^2 .* squeeze(LFWarpmask(t,s,:,:,:));
            WarpMaskSum = WarpMaskSum + squeeze(sum(sum(LFWarpmask, 1), 2));
        end
    end
end

% Compute mean of squared variance
LFWarpVar2 = LFWarpVar2 ./ WarpMaskSum;

LFWarpVar2mask = ~isnan(LFWarpVar2);
LFWarpVar2(~LFWarpVar2mask) = 0;
LFWarpVar2Mean = sum(LFWarpVar2(:)) ./ sum(double(LFWarpVar2mask(:)));

% Final metric is similar to PSNR;
LFAC = 10 * log10(255*255 / LFWarpVar2Mean);

end