function opLab = genLabels(ipLab, nF, wd, sigma)
opLab = zeros(2,nF);
% Ignore dim 1 for now
opLab(2,ipLab(:,1)) = 1;
gwin = gausswin(wd,sigma);     % Hardcoded, needs to be handled better
opLab(2,:) = conv(opLab(2,:),gwin,'same');

% Make sure the labels at each frame sum to one
opLab(2,:) = opLab(2,:)/max(opLab(2,:));
opLab(1,:) = 1 - opLab(2,:);

