from ARIO_modify import *
import pandas as pd


class FinalExps:
    def __init__(self, I_fraction_threshold=1e-3, Re_thre=1, phi=0.01, k=100, num_time_steps_in_inv_init=3):
        self.eco_epi_model = ARIO(I_fraction_threshold=I_fraction_threshold,
                                  Re_thre=Re_thre, phi=phi, k=k,
                                  num_time_steps_in_inv_init=num_time_steps_in_inv_init)
        self.economic_region_npi_policy, self.economic_region_chi_value = \
            self.eco_epi_model.epidemic_model.define_country_npi_policy_class_in_economic()

    def main_results(self, rho, varphi, NPI_policy_scenarios, keep_time_steps):
        main_results_file = self.eco_epi_model.hyper_matlab_results_path + '/' \
                            + 'main_rho_' + str(rho) + '_varphi_' + str(varphi)
        if not os.path.exists(main_results_file):
            os.mkdir(main_results_file)

        results = {}
        for time_index in range(len(NPI_policy_scenarios)):
            NPI_policy_scenario, keep_time_step = NPI_policy_scenarios[time_index], keep_time_steps[time_index]
            scenario_key, eco_epi_data = self.eco_epi_model.load_economic_data(rho, varphi, NPI_policy_scenario,
                                                                               keep_time_step,
                                                                               if_check_economic_data=True)
            prevalence = eco_epi_data['epidemic_data']['I_frac']
            cum_mortality_rate = eco_epi_data['epidemic_data']['D_frac']

            value_added_actual_by_country = \
                self.eco_epi_model.cal_sum_by_regions_for_time_series_arr(eco_epi_data['value_added_actual'])
            value_added_without_evo_by_country = \
                self.eco_epi_model.cal_sum_by_regions_for_time_series_arr(
                    np.array(eco_epi_data['value_added_without_evo']))
            value_added_max_by_country = \
                self.eco_epi_model.cal_sum_by_regions_for_time_series_arr(np.array(eco_epi_data['value_added_max']))
            value_added_change_due_to_evo_by_country = value_added_max_by_country - value_added_without_evo_by_country
            value_added_original_by_country = self.eco_epi_model.cal_sum_by_regions_for_time_series_arr(
                np.array(eco_epi_data['value_added_original'])[None, :]).flatten()

            for country_class in self.economic_region_npi_policy:
                corr_regions_index_list = self.economic_region_npi_policy[country_class]
                results[scenario_key + '_prev_' + country_class] = \
                    np.mean(prevalence[:, corr_regions_index_list], axis=1)
                results[scenario_key + '_cum_mor_' + country_class] = \
                    np.mean(cum_mortality_rate[:, corr_regions_index_list], axis=1)

                # ----- total ----------
                results[scenario_key + '_ave_tot_GVA_' + country_class] = \
                    np.mean(value_added_actual_by_country[:, corr_regions_index_list] /
                            value_added_original_by_country[corr_regions_index_list][None, :], axis=1) - 1
                results[scenario_key + '_tot_GVA_' + country_class] = \
                    np.sum(value_added_actual_by_country[:, corr_regions_index_list], axis=1) / \
                    sum(value_added_original_by_country[corr_regions_index_list]) - 1

                # ----- direct ----------
                results[scenario_key + '_ave_di_GVA_' + country_class] = \
                    np.mean(value_added_without_evo_by_country[:, corr_regions_index_list] /
                            value_added_original_by_country[corr_regions_index_list][None, :], axis=1) - 1
                results[scenario_key + '_di_GVA_' + country_class] = \
                    np.sum(value_added_without_evo_by_country[:, corr_regions_index_list], axis=1) / \
                    sum(value_added_original_by_country[corr_regions_index_list]) - 1

                # ----- indirect by evo ----------
                results[scenario_key + '_ave_ind_evoGVA_' + country_class] = \
                    np.mean(value_added_change_due_to_evo_by_country[:, corr_regions_index_list] /
                            value_added_original_by_country[corr_regions_index_list][None, :], axis=1)
                results[scenario_key + '_ind_evoGVA_' + country_class] = \
                    np.sum(value_added_change_due_to_evo_by_country[:, corr_regions_index_list], axis=1) / \
                    sum(value_added_original_by_country[corr_regions_index_list])

                # ----- indirect by propagation ----------
                results[scenario_key + '_ave_ind_propGVA_' + country_class] = \
                    results[scenario_key + '_ave_tot_GVA_' + country_class] - \
                    results[scenario_key + '_ave_di_GVA_' + country_class] - \
                    results[scenario_key + '_ave_ind_evoGVA_' + country_class]
                results[scenario_key + '_ind_propGVA_' + country_class] = \
                    results[scenario_key + '_tot_GVA_' + country_class] - \
                    results[scenario_key + '_di_GVA_' + country_class] - \
                    results[scenario_key + '_ind_evoGVA_' + country_class]
        results_df = pd.DataFrame(results)
        results_df.to_csv(main_results_file + '/main_results.csv', index=False)

    def market_structure_change(self, rho, varphi, NPI_policy_scenarios, keep_time_steps):
        main_results_file = self.eco_epi_model.hyper_matlab_results_path + '/' + 'main_rho_' \
                            + str(rho) + '_varphi_' + str(varphi)

        full_market_share_change_results_save_path = main_results_file + '/full_market_change.xlsx'
        all_comms_df = pd.DataFrame([comm.upper() for comm in self.eco_epi_model.activities])
        all_comms_df.to_excel(full_market_share_change_results_save_path,
                              sheet_name='heat_x_labels', index=False, header=None)

        all_regions_df = pd.DataFrame([reg.upper() for reg in self.eco_epi_model.regions])
        with pd.ExcelWriter(full_market_share_change_results_save_path, engine="openpyxl", mode='a') as writer:
            all_regions_df.to_excel(writer, sheet_name='heat_y_labels', index=False, header=None)

        market_share_change_results_save_path = main_results_file + '/market_change.xlsx'
        market_share_change_results = {}

        target_sector = [
            'pdr',
            'rmk',
            'afs',
            'tex',
            'eeq',
            'coa',
            'trd',
            'edu',
            'gdt',
            'osg',
        ]
        reshape_init_x = np.reshape(self.eco_epi_model.x, (self.eco_epi_model.num_regions, -1))
        target_region_index = \
            np.argsort(np.sum(reshape_init_x, axis=1))[::-1][:10]
        target_region = [self.eco_epi_model.regions[index] for index in target_region_index]
        min_value_in_heatmap, max_value_in_heatmap = np.inf, -np.inf
        example_sector = ['pdr',
                          'eeq',
                          'tex',
                          'edu',
                          ]
        regions_CHI_class = {}
        items_with_largest_three_regions_in_example_sector = []
        for sector in example_sector:
            example_sector_index = self.eco_epi_model.activities.index(sector)
            for region_index in np.argsort(reshape_init_x[:, example_sector_index])[::-1][:3]:
                regions_CHI_class[self.eco_epi_model.regions[region_index]] = \
                    self.eco_epi_model.get_eco_region_CHI_class(self.economic_region_npi_policy, region_index)
                items_with_largest_three_regions_in_example_sector.append(
                    self.eco_epi_model.regions[region_index] + '_' + sector)

        for time_index in range(len(NPI_policy_scenarios)):
            NPI_policy_scenario, keep_time_step = NPI_policy_scenarios[time_index], keep_time_steps[time_index]
            target_sectors_market_structure_change = []
            scenario_key, eco_epi_data = self.eco_epi_model.load_economic_data(rho, varphi, NPI_policy_scenario,
                                                                          keep_time_step, if_check_economic_data=True)

            init_production_distribution = eco_epi_data['production_distribution'][0]
            final_production_distribution = eco_epi_data['production_distribution'][-1]
            sector_distribution_change = np.sum(
                np.reshape(np.power(final_production_distribution - init_production_distribution, 2),
                           (self.eco_epi_model.num_regions, -1)), axis=0)
            product_distribution_change = \
                np.reshape(self.eco_epi_model.division_define(
                    final_production_distribution - init_production_distribution, init_production_distribution, 0),
                    (self.eco_epi_model.num_regions, -1))
            needed_production_distribution_change = \
                product_distribution_change[target_region_index][:, [self.eco_epi_model.activities.index(sector)
                                                                     for sector in target_sector]]
            df = pd.DataFrame(needed_production_distribution_change)
            if time_index > 0:
                with pd.ExcelWriter(market_share_change_results_save_path, engine="openpyxl", mode='a') as writer:
                    df.to_excel(writer, sheet_name=scenario_key + '_heat', index=False, header=None)
            else:
                df.to_excel(market_share_change_results_save_path, sheet_name=scenario_key + '_heat',
                            index=False, header=None)

            last_day_in_epidemic_control = \
                self.eco_epi_model.get_last_day_in_epidemic_control_in_eco_time_step(
                    NPI_policy_scenario, keep_time_step)
            reopen_time_production_distribution = eco_epi_data['production_distribution'][last_day_in_epidemic_control]
            reopen_time_product_distribution_change = \
                np.reshape(self.eco_epi_model.division_define(
                    reopen_time_production_distribution - init_production_distribution, init_production_distribution,
                    0),
                    (self.eco_epi_model.num_regions, -1))
            reopen_time_needed_production_distribution_change = \
                reopen_time_product_distribution_change[target_region_index][:,
                [self.eco_epi_model.activities.index(sector) for sector in target_sector]]

            for sector in target_sector:
                target_sectors_market_structure_change.append(
                    sector_distribution_change[self.eco_epi_model.activities.index(sector)])

            df = pd.DataFrame(target_sectors_market_structure_change)
            with pd.ExcelWriter(market_share_change_results_save_path, engine="openpyxl", mode='a') as writer:
                df.to_excel(writer, sheet_name=scenario_key + '_mar', index=False, header=None)

            df = pd.DataFrame(reopen_time_needed_production_distribution_change)
            with pd.ExcelWriter(market_share_change_results_save_path, engine="openpyxl", mode='a') as writer:
                df.to_excel(writer, sheet_name=scenario_key + '_re_heat', index=False, header=None)

            min_value_in_heatmap, max_value_in_heatmap = min(min_value_in_heatmap,
                                                             np.min(needed_production_distribution_change)), \
                                                         max(max_value_in_heatmap,
                                                             np.max(needed_production_distribution_change))

            for example_item in items_with_largest_three_regions_in_example_sector:
                market_share_change_results[scenario_key + '_' + example_item] = \
                    np.array(eco_epi_data['production_distribution'])[:,
                    list(self.eco_epi_model.item_info).index(example_item)]

            temp_market_share_change_results = np.reshape(product_distribution_change,
                                                          (self.eco_epi_model.num_regions, -1))
            df = pd.DataFrame(temp_market_share_change_results)
            with pd.ExcelWriter(full_market_share_change_results_save_path, engine="openpyxl", mode='a') as writer:
                df.to_excel(writer, sheet_name=scenario_key + '_heat', index=False, header=None)

            reopen_time_temp_market_share_change_results = np.reshape(reopen_time_product_distribution_change,
                                                                      (self.eco_epi_model.num_regions, -1))
            df = pd.DataFrame(reopen_time_temp_market_share_change_results)
            with pd.ExcelWriter(full_market_share_change_results_save_path, engine="openpyxl", mode='a') as writer:
                df.to_excel(writer, sheet_name=scenario_key + '_re_heat', index=False, header=None)

        df = pd.DataFrame([min_value_in_heatmap, max_value_in_heatmap])
        with pd.ExcelWriter(market_share_change_results_save_path, engine="openpyxl", mode='a') as writer:
            df.to_excel(writer, sheet_name='heat_min_max', index=False, header=None)

        df = pd.DataFrame([sector.upper() for sector in target_sector])
        with pd.ExcelWriter(market_share_change_results_save_path, engine="openpyxl", mode='a') as writer:
            df.to_excel(writer, sheet_name='heat_x_labels', index=False, header=None)

        df = pd.DataFrame([region.upper() for region in target_region])
        with pd.ExcelWriter(market_share_change_results_save_path, engine="openpyxl", mode='a') as writer:
            df.to_excel(writer, sheet_name='heat_y_labels', index=False, header=None)


        df = pd.DataFrame(market_share_change_results)
        with pd.ExcelWriter(market_share_change_results_save_path, engine="openpyxl", mode='a') as writer:
            df.to_excel(writer, sheet_name='market_share_example', index=False)
        df = pd.DataFrame(regions_CHI_class, index=[0]).T
        with pd.ExcelWriter(market_share_change_results_save_path, engine="openpyxl", mode='a') as writer:
            df.to_excel(writer, sheet_name='regions_CHI_class')

    def competition_explain(self, rho, varphi, NPI_policy_scenarios, keep_time_steps):
        main_results_file = self.eco_epi_model.hyper_matlab_results_path + '/' + 'main_rho_' + str(rho) + '_varphi_' \
                            + str(varphi)
        results_save_path = main_results_file + '/competition_explain.xlsx'

        reshape_init_x = np.reshape(self.eco_epi_model.x, (self.eco_epi_model.num_regions, -1))
        sector_list = ['eeq', 'edu', 'tex', 'coa', 'trd', 'gdt']
        competition_results = {}
        init_x_info = {}
        vartheta_info = {}
        corr_max_producer_name = {}
        CHI_scores = {}
        CHI_levels = {}
        for time_index in range(len(NPI_policy_scenarios)):
            NPI_policy_scenario, keep_time_step = NPI_policy_scenarios[time_index], keep_time_steps[time_index]
            scenario_key, eco_epi_data = self.eco_epi_model.load_economic_data(rho, varphi, NPI_policy_scenario,
                                                                          keep_time_step, if_check_economic_data=True)
            for sec in sector_list:
                sec_index = self.eco_epi_model.activities.index(sec)

                sector_ave_profit = \
                    np.sum(np.array(eco_epi_data['profit'])[:, self.eco_epi_model.product_index_list_in_items(sec_index)] *
                           np.array(eco_epi_data['production_distribution'])[:,
                           self.eco_epi_model.product_index_list_in_items(sec_index)], axis=1)

                target_region_list_index = np.argsort(reshape_init_x[:, sec_index])[::-1][:5]  # 产量前5的地方
                if keep_time_step == 52 * 7 * 1:
                    init_x_info[sec] = 100 * reshape_init_x[target_region_list_index, sec_index] / sum(
                        reshape_init_x[:, sec_index])
                    vartheta_info[sec] = [self.eco_epi_model.NPI_sector_multiplier[sec_index],
                                          self.eco_epi_model.demand_multiplier_sector[sec_index]]
                    corr_max_producer_name[sec] = [self.eco_epi_model.regions[reg_index].upper() for reg_index in
                                                   target_region_list_index]
                    CHI_scores[sec] = [self.economic_region_chi_value[self.eco_epi_model.regions[reg_index].upper()] for reg_index
                                       in target_region_list_index]
                    CHI_levels[sec] = [self.eco_epi_model.get_eco_region_CHI_class(self.economic_region_npi_policy, reg_index) for
                                       reg_index in target_region_list_index]
                for rank, reg_index in enumerate(target_region_list_index):
                    item_index = reg_index * 65 + sec_index
                    reliability = np.array(eco_epi_data['reliability'])[:, item_index]
                    kappa = np.array(eco_epi_data['kappa'])[:, item_index]
                    profit = np.array(eco_epi_data['profit'])[:, item_index]
                    var_profit = np.array(eco_epi_data['var_profit'])[:, item_index]
                    total_order = np.array(eco_epi_data['total_order'])[:, item_index]

                    competition_results[scenario_key + '_' + sec + '_' + str(rank) + 're'] = reliability
                    competition_results[scenario_key + '_' + sec + '_' + str(rank) + 'ka'] = kappa
                    competition_results[scenario_key + '_' + sec + '_' + str(rank) + 'pro'] = profit
                    competition_results[scenario_key + '_' + sec + '_' + str(rank) + 'sec_pro'] = sector_ave_profit
                    competition_results[scenario_key + '_' + sec + '_' + str(rank) + 'var_pro'] = var_profit
                    competition_results[scenario_key + '_' + sec + '_' + str(rank) + 'or'] = total_order

        df = pd.DataFrame(competition_results)
        df.to_excel(results_save_path, sheet_name='results', index=False)
        df = pd.DataFrame(init_x_info)
        with pd.ExcelWriter(results_save_path, engine="openpyxl", mode='a') as writer:
            df.to_excel(writer, sheet_name='init_x_info', index=False)
        df = pd.DataFrame(CHI_scores)
        with pd.ExcelWriter(results_save_path, engine="openpyxl", mode='a') as writer:
            df.to_excel(writer, sheet_name='CHI_scores', index=False)
        df = pd.DataFrame(CHI_levels)
        with pd.ExcelWriter(results_save_path, engine="openpyxl", mode='a') as writer:
            df.to_excel(writer, sheet_name='CHI_levels', index=False)
        df = pd.DataFrame(corr_max_producer_name)
        with pd.ExcelWriter(results_save_path, engine="openpyxl", mode='a') as writer:
            df.to_excel(writer, sheet_name='large_reg', index=False)
        df = pd.DataFrame(vartheta_info)
        with pd.ExcelWriter(results_save_path, engine="openpyxl", mode='a') as writer:
            df.to_excel(writer, sheet_name='vartheta', index=False)

    def all_experiments(self, rho, varphi, NPI_policy_scenarios, keep_time_steps):
        self.main_results(rho=rho, varphi=varphi, NPI_policy_scenarios=NPI_policy_scenarios,
                          keep_time_steps=keep_time_steps)
        self.market_structure_change(rho=rho, varphi=varphi, NPI_policy_scenarios=NPI_policy_scenarios,
                                     keep_time_steps=keep_time_steps)
        self.competition_explain(rho=rho, varphi=varphi, NPI_policy_scenarios=NPI_policy_scenarios,
                                 keep_time_steps=keep_time_steps)


# main results
main_exps = FinalExps()
main_exps.all_experiments(rho=0.04, varphi=0.8,
                          NPI_policy_scenarios=['keep_curr', 'linear', 'quad_neg'],
                          keep_time_steps=[52 * 7 * 1.0, 52 * 7 * 1.5, 52 * 7 * 1.5])

#### supplementary results
# ### change rho
# rho=0.025
exp_supple_change_rho = FinalExps()
exp_supple_change_rho.main_results(rho=0.025, varphi=0.8,
                                   NPI_policy_scenarios=['keep_curr', 'linear', 'quad_neg'],
                                   keep_time_steps=[52 * 7 * 1.0, 52 * 7 * 1.5, 52 * 7 * 1.5])

# rho=0.03
exp_supple_change_rho = FinalExps()
exp_supple_change_rho.main_results(rho=0.03, varphi=0.8,
                                   NPI_policy_scenarios=['keep_curr', 'linear', 'quad_neg'],
                                   keep_time_steps=[52 * 7 * 1.0, 52 * 7 * 1.5, 52 * 7 * 1.5])

# rho=0.045
exp_supple_change_rho = FinalExps()
exp_supple_change_rho.main_results(rho=0.045, varphi=0.8,
                                   NPI_policy_scenarios=['keep_curr', 'linear', 'quad_neg'],
                                   keep_time_steps=[52 * 7 * 1.0, 52 * 7 * 1.5, 52 * 7 * 1.5])

# rho=0.05
exp_supple_change_rho = FinalExps()
exp_supple_change_rho.main_results(rho=0.05, varphi=0.8,
                                   NPI_policy_scenarios=['keep_curr', 'linear', 'quad_neg'],
                                   keep_time_steps=[52 * 7 * 1.0, 52 * 7 * 1.5, 52 * 7 * 1.5])

# rho=0.06
exp_supple_change_rho = FinalExps()
exp_supple_change_rho.main_results(rho=0.06, varphi=0.8,
                                   NPI_policy_scenarios=['keep_curr', 'linear', 'quad_neg'],
                                   keep_time_steps=[52 * 7 * 1.0, 52 * 7 * 1.5, 52 * 7 * 1.5])

# ### change varphi
# varphi=0.3
exp_supple_change_varphi = FinalExps()
exp_supple_change_varphi.main_results(rho=0.04, varphi=0.3,
                                   NPI_policy_scenarios=['keep_curr', 'linear', 'quad_neg'],
                                   keep_time_steps=[52 * 7 * 1.0, 52 * 7 * 1.5, 52 * 7 * 1.5])
# varphi=0.5
exp_supple_change_varphi = FinalExps()
exp_supple_change_varphi.main_results(rho=0.04, varphi=0.5,
                                   NPI_policy_scenarios=['keep_curr', 'linear', 'quad_neg'],
                                   keep_time_steps=[52 * 7 * 1.0, 52 * 7 * 1.5, 52 * 7 * 1.5])
# varphi=0.7
exp_supple_change_varphi = FinalExps()
exp_supple_change_varphi.main_results(rho=0.04, varphi=0.7,
                                   NPI_policy_scenarios=['keep_curr', 'linear', 'quad_neg'],
                                   keep_time_steps=[52 * 7 * 1.0, 52 * 7 * 1.5, 52 * 7 * 1.5])
# varphi=0.9
exp_supple_change_varphi = FinalExps()
exp_supple_change_varphi.main_results(rho=0.04, varphi=0.9,
                                   NPI_policy_scenarios=['keep_curr', 'linear', 'quad_neg'],
                                   keep_time_steps=[52 * 7 * 1.0, 52 * 7 * 1.5, 52 * 7 * 1.5])
# #
# ### change phi
# phi=0.005
exp_supple_change_phi = FinalExps(phi=0.005)
exp_supple_change_phi.main_results(rho=0.04, varphi=0.8,
                                   NPI_policy_scenarios=['keep_curr', 'linear', 'quad_neg'],
                                   keep_time_steps=[52 * 7 * 1.0, 52 * 7 * 1.5, 52 * 7 * 1.5])

### change I_threshold
exp_supple_change_I_fraction_threshold = FinalExps(I_fraction_threshold=1e-1)
exp_supple_change_I_fraction_threshold.main_results(rho=0.04, varphi=0.8,
                                   NPI_policy_scenarios=['keep_curr', 'linear', 'quad_neg'],
                                   keep_time_steps=[52 * 7 * 1.0, 52 * 7 * 1.5, 52 * 7 * 1.5])

