function[frssl] = smooth1cell(fr)
try
temp=fr(end,:);
catch
 temp=nan;   
end
 fr=fr(1:end-1,:);

hf=fspecial('gaussian',[3 3],.8);

for t=1:size(fr,2)
     
        smoo(:,:,t)=(conv2(reshape(fr(:,t),[sqrt(size(fr,1)) sqrt(size(fr,1))]),hf,'same'));
     
end

%% 2. sparse:
e = 0.5*mad(smoo(:));
spa=sign(smoo(:)).*(max(abs(smoo(:)) - e,0));
spa=reshape(spa,size(fr));


    %% 3 .low rank using SVD
    spa(isnan(spa))=0;

    W = reshape(spa,[size(fr,1) size(spa,2)]);
    [U,S,V]=svd(W);
    dS = diag(S);
    try    e = dS(5);
    catch
        e = dS(4);
    end
    %     dSp = sign(dS).*max(abs(dS)-e,0);
     S(S<e)=0;
    Wst(:,:) = U*S*V';

frssl=Wst;reshape(Wst,[sqrt(size(fr,1)) sqrt(size(fr,1)) size(Wst,2)]);

frssl(end+1,:)= temp;

end