function [LFEC, LFWarpVar2, LFWarp, LFWarpmask] = LightFieldEpipolarConsistency(LF, dispMap, interp)

if nargin < 3
    interp = 'linear';
end

rows = size(LF,1);
cols = size(LF,2);
height = size(LF,3);
width = size(LF,4);
nch = size(LF,5);

sc = ceil(cols/2);
tc = ceil(rows/2);
LFWarp = zeros(size(LF));
LFWarp(tc,sc,:,:,:) = squeeze(LF(tc,sc,:,:,:));
LFWarpmask = zeros(size(LF));
LFWarpmask(tc,sc,:,:,:) = ones(height, width, 3);

if length(size(dispMap)) == 4 % Disparity map for every view
    % Back-warping to center view from center disparity map
    imWhite = ones(height, width, nch);
    for t = 1:rows
        for s = 1:cols
            if t == tc && s == sc, continue, end
            ds = sc - s;
            dt = tc - t;
            dispst = squeeze(dispMap(t,s,:,:));
            Xwarp = dispst * ds;
            Ywarp = dispst * dt;

%             for ch = 1:nch
%                 LFWarp(t,s,:,:,ch) = imwarp(squeeze(LF(t,s,:,:,ch)),cat(3,Xwarp,Ywarp), interp);
%             end
%             stmask = imwarp(imWhite,cat(3,Xwarp,Ywarp), interp);
%             LFWarpmask(t,s,:,:,:) = cat(3,stmask,stmask,stmask);
            LFWarp(t,s,:,:,:) = imwarp(squeeze(LF(t,s,:,:,:)),cat(3,Xwarp,Ywarp), interp);
            LFWarpmask(t,s,:,:,:) = imwarp(imWhite,cat(3,Xwarp,Ywarp), interp);
        end
    end
else % Disparity map for centre view only
    % Back-warping to center view from center disparity map
    [X, Y] = meshgrid(1:width, 1:height);
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
end

% Compute back warp views variance map
LFWarpMean = squeeze(sum(sum(LFWarp .* LFWarpmask, 1), 2) ./ sum(sum(LFWarpmask, 1), 2));
LFWarpVar2 = zeros(height, width, 3);
for t = 1:rows
    for s = 1:cols
        LFWarpVar2 = LFWarpVar2 + (squeeze(LFWarp(t,s,:,:,:)) - LFWarpMean).^2 .* squeeze(LFWarpmask(t,s,:,:,:));
    end
end
LFWarpVar2 = LFWarpVar2 ./ squeeze(sum(sum(LFWarpmask, 1), 2));

LFWarpVar2mask = ~isnan(LFWarpVar2);
LFWarpVar2(~LFWarpVar2mask) = 0;
LFWarpVar2Mean = sum(LFWarpVar2(:)) ./ sum(double(LFWarpVar2mask(:)));

% Final metric is similar to PSNR;
LFEC = 10 * log10(255*255 / LFWarpVar2Mean);

end