beta_old = parametersLPP(:, 2, 1);
beta_sindy = parametersLPP(:, 5, 1);
beta_full_sindy = parametersLPP(:, 6, 1);

phi_old = parametersLPP(:, 2, 5);
phi_sindy = parametersLPP(:, 5, 5);
phi_full_sindy = parametersLPP(:, 6, 6);

lr1_old = parametersLPP(:, 2, 2);
lr2_old = parametersLPP(:, 2, 3);
rew_coef_sindy = parametersLPP(:, 5, 2);
qval_coef_sindy = parametersLPP(:, 5, 3);
rew_coef_full_sindy = parametersLPP(:, 6, 2);
qval_coef_full_sindy = parametersLPP(:, 6, 3);

n = length(beta_old);
x = [ones(n,1), 2*ones(n,1), 3*ones(n,1)];
y = [abs(phi_old)-beta_old, abs(phi_sindy)-beta_sindy, abs(phi_full_sindy)-beta_full_sindy];
subplot(2,1,1);
scatter(x, y, [],'blue','filled','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.005);
hold on;
scatter([1, 2, 3], [mean(abs(phi_old)-beta_old), mean(abs(phi_sindy)-beta_sindy), mean(abs(phi_full_sindy)-beta_full_sindy)], [], 'red', 'filled');
err = [std(abs(phi_old)-beta_old)/sqrt(length(beta_old)), std(abs(phi_sindy)-beta_sindy)/sqrt(length(beta_old)), std(abs(phi_full_sindy)-beta_full_sindy)/sqrt(length(beta_old))];
errorbar([1, 2, 3], [mean(abs(phi_old)-beta_old), mean(abs(phi_sindy)-beta_sindy), mean(abs(phi_full_sindy)-beta_full_sindy)],err, 'LineStyle', 'none', 'Color', 'red');
xlim([.5, 3.5]);
ylabel('abs(\phi)-\beta');
names = {'Classic Full'; 'SINDy Simple'; 'SINDy Full'};
set(gca,'xtick',[1:3],'xticklabel',names);
title({sprintf('Classic Full BIC: %f', biclist(2, 1)); sprintf('SINDy Simple BIC: %f', biclist(5, 1)); sprintf('SINDy Full BIC: %f', biclist(6, 1))});

x = [ones(n,1), 2*ones(n,1), 3*ones(n,1)];
y = [lr1_old-lr2_old, rew_coef_sindy-qval_coef_sindy, rew_coef_full_sindy-qval_coef_full_sindy];
subplot(2,1,2);
scatter(x, y, [], 'blue','filled','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.005);
hold on;
scatter([1, 2, 3], [mean(lr1_old-lr2_old), mean(rew_coef_sindy-qval_coef_sindy), mean(rew_coef_full_sindy-qval_coef_full_sindy)], [], 'red', 'filled');
err = [std(lr1_old-lr2_old)/sqrt(length(beta_old)), std(rew_coef_sindy-qval_coef_sindy)/sqrt(length(beta_old)), std(rew_coef_full_sindy-qval_coef_full_sindy)/sqrt(length(beta_old))];
errorbar([1, 2, 3], [mean(lr1_old-lr2_old), mean(rew_coef_sindy-qval_coef_sindy), mean(rew_coef_full_sindy-qval_coef_full_sindy)],err, 'LineStyle', 'none', 'Color', 'red');
xlim([.5, 3.5])
ylabel({'Learning bias (\alpha_{CONF} - \alpha_{DIS})';'OR';'Optimism (c_{reward} - c_{qval})'});
names = {'Classic Full'; 'SINDy Simple'; 'SINDy Full'};
set(gca,'xtick',[1:3],'xticklabel',names);
title('');