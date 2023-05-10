
% ------- baseline --------------------
raw_epidemic_data_info = '2022-03-01_IP_7_2022-11-11';
I_thre = '0.001';
Re_thre = '1';
phi = '0.01';
k = '100';
CHI_thre = '25_50';
rho = '0.04';
varphi = '0.8';
NPI_policy_scenario = 'keep_curr_';
% -------------------------------------

% main text
epi_draw(raw_epidemic_data_info, I_thre, Re_thre, phi, k, CHI_thre, rho, ...
    varphi, NPI_policy_scenario,  [0 4], [0.75 1.15], [0.75 0.95 1.15], true, '')
eco_draw_add(raw_epidemic_data_info, I_thre, Re_thre, phi, k, CHI_thre, rho, ...
    varphi, NPI_policy_scenario, [-62, -62, -62, -62], [20, 20, 20, 20], true, '', [0.15, 0.83, 0.08, 0.05])
com_explain(raw_epidemic_data_info, I_thre, Re_thre, phi, k, CHI_thre, rho, ...
    varphi, NPI_policy_scenario, true, '')
sector_change_modify(raw_epidemic_data_info, I_thre, Re_thre, phi, k, CHI_thre, rho, ...
    varphi, NPI_policy_scenario, true, '')
% 
% 
% % % supple only for base
epi_draw(raw_epidemic_data_info, I_thre, Re_thre, phi, k, CHI_thre, rho, ...
    varphi, NPI_policy_scenario,  [0 15], [0.05 1.15], [0.05 0.6 1.15], false, '')
sector_change_all(raw_epidemic_data_info, I_thre, Re_thre, phi, k, CHI_thre, rho, varphi, NPI_policy_scenario)

% % supple sensitivity analysis
% % % change rho
eco_draw(raw_epidemic_data_info, I_thre, Re_thre, phi, k, CHI_thre, '0.025', ...
     varphi, NPI_policy_scenario, [-62, -62, -62, -62], [20, 20, 20, 20], false, 'rho_sensi_0.025', [0.19, 0.82, 0.08, 0.05])
 eco_draw(raw_epidemic_data_info, I_thre, Re_thre, phi, k, CHI_thre, '0.03', ...
     varphi, NPI_policy_scenario, [-62, -62, -62, -62], [20, 20, 20, 20], false, 'rho_sensi_0.03', [0.19, 0.82, 0.08, 0.05])
eco_draw(raw_epidemic_data_info, I_thre, Re_thre, phi, k, CHI_thre, '0.06', ...
     varphi, NPI_policy_scenario, [-62, -62, -62, -62], [20, 20, 20, 20], false, 'rho_sensi_0.06', [0.19, 0.82, 0.08, 0.05])

% % % change varphi
eco_draw(raw_epidemic_data_info, I_thre, Re_thre, phi, k, CHI_thre, rho, ...
     '0.5', NPI_policy_scenario, [-62, -62, -62, -62], [20, 20, 20, 20], false, 'varphi_sensi_0.5', [0.19, 0.82, 0.08, 0.05])
eco_draw(raw_epidemic_data_info, I_thre, Re_thre, phi, k, CHI_thre, rho, ...
    '0.9', NPI_policy_scenario,[-62, -62, -62, -62], [20, 20, 20, 20], false, 'varphi_sensi_0.9', [0.19, 0.82, 0.08, 0.05])
% change phi
epi_draw(raw_epidemic_data_info, I_thre, Re_thre, '0.005', k, CHI_thre, rho, ...
       varphi, NPI_policy_scenario,  [0 15], [0 1.5], [0 0.5 1 1.5], false, 'phi_sensi_0.005')
eco_draw(raw_epidemic_data_info, I_thre, Re_thre, '0.005', k, CHI_thre, rho, ...
     varphi, NPI_policy_scenario, [-62, -62, -62, -62], [20, 20, 20, 20], false, 'phi_sensi_0.005', [0.19, 0.82, 0.08, 0.05])
% change I_thre
eco_draw(raw_epidemic_data_info, '0.1', Re_thre, phi, k, CHI_thre, rho, ...
     varphi, NPI_policy_scenario, [-62, -62, -62, -62], [20, 20, 20, 20], false, 'I_thre_sensi_0.1', [0.19, 0.82, 0.08, 0.05])