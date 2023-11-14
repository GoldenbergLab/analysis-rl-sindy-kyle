%% calculate BIC and AIC (added by Kyle)
deckerfit = readtable('..\output\potter_data\RL\potter_fits.txt');
ages = readtable('..\data\potter\potter_ages.csv');
subjecttot = 74; %80 in decker, 74 in potter, 151 in online
n_model = 1;

bic = NaN(subjecttot, n_model); biclist = NaN(n_model,3); bicadultlist = NaN(n_model,3);
aic = NaN(subjecttot, n_model); aiclist = NaN(n_model,3);
bic_adult = [];
nfpm = 6; %6 in classic, 7 in sindy
lik_col = 8; %8 in classic, 9 in sindy
id_col = lik_col+1;
age_col = 2;

if subjecttot == 151
    deckerfit.("Var10") = string(deckerfit.("Var10"));
    ages.("subject_id") = string(ages.("subject_id"));
    age_col = 3;
end

for model = 1:n_model
    for sub = 1:subjecttot
        bic(sub, model) = log(189)*nfpm+2*deckerfit{sub,lik_col};
        aic(sub, model) = 2*nfpm-2*deckerfit{sub,lik_col};
        if subjecttot == 151
            id = deckerfit{sub,id_col}{1};
        else
            id = deckerfit{sub,id_col};
        end
        if ages{(ages.subject_id==id),age_col} >=18
            bic_adult(length(bic_adult)+1) = log(189)*nfpm-2*deckerfit{sub,lik_col};
        end

    end
    biclist(model,1) = sum(bic(:,model));
    aiclist(model,1) = sum(aic(:,model));
    biclist(model,2) = mean(bic(:,model));
    aiclist(model,2) = mean(aic(:,model));
    biclist(model,3) = std(bic(:,model)) / sqrt(length(bic(:,model)));
    aiclist(model,3) = std(aic(:,model)) / sqrt(length(aic(:,model)));

    bicadultlist(model,1) = sum(bic_adult(model,:));
    bicadultlist(model,2) = mean(bic_adult(model,:));
    bicadultlist(model,3) = std(bic_adult(model,:)) / sqrt(length(bic_adult(model,:)));

end

