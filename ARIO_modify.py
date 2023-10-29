from SVEIRD_meta import *
import eco_epi_hyper

class ARIO:
    def __init__(self, I_fraction_threshold=1e-3, Re_thre=1, phi=0.01, k=100, num_time_steps_in_inv_init=3):

        hyper_paras = eco_epi_hyper.load_hyper()
        self.hyper_file_path_main = 'results_for_economic_cal/' + \
                                    hyper_paras['t0_date_owid'] + '_IP_' + str(hyper_paras['active_case_cal_length'])
        self.hyper_matlab_results_path_main = \
            'matlab_files/results/' + \
            hyper_paras['t0_date_owid'] + '_IP_' + str(hyper_paras['active_case_cal_length'])
        if not os.path.exists(self.hyper_file_path_main):
            os.mkdir(self.hyper_file_path_main)
        if not os.path.exists(self.hyper_matlab_results_path_main):
            os.mkdir(self.hyper_matlab_results_path_main)
        self.hyper_file_path = \
            self.hyper_file_path_main + '/' + \
            'I_thre_' + str(I_fraction_threshold) + \
            '_Re_thre_' + str(Re_thre) + '_phi_' + str(phi) + \
            '_k_' + str(k) + '_CHI_thre_' + \
            str(hyper_paras['CHI_thre_mild']) + '_' + str(hyper_paras['CHI_thre_moderate'])
        self.hyper_matlab_results_path = \
            self.hyper_matlab_results_path_main + '/' + \
            'I_thre_' + str(I_fraction_threshold) + \
            '_Re_thre_' + str(Re_thre) + '_phi_' + str(phi) + \
            '_k_' + str(k) + '_CHI_thre_' + \
            str(hyper_paras['CHI_thre_mild']) + '_' + str(hyper_paras['CHI_thre_moderate'])
        if not os.path.exists(self.hyper_file_path):
            os.mkdir(self.hyper_file_path)
        if not os.path.exists(self.hyper_matlab_results_path):
            os.mkdir(self.hyper_matlab_results_path)

        self.I_fraction_threshold = I_fraction_threshold
        self.num_time_steps_in_inv_init = num_time_steps_in_inv_init
        self.num_time_steps_in_target_inventory = self.num_time_steps_in_inv_init + 1
        self.epidemic_model = DiscreteSVEIRD(Re_thre=Re_thre, phi=phi, k=k, hyper_file_path=self.hyper_file_path)
        self.days_in_each_eco_time_step = hyper_paras['days_in_each_eco_time_step']
        self.T = hyper_paras['T']
        self.regions = hyper_paras['regions']
        self.activities = hyper_paras['activities']
        self.NPI_sector_multiplier = np.array(hyper_paras['NPI_sector_multiplier'])
        self.demand_multiplier_sector = np.array(hyper_paras['demand_multiplier_sector'])
        self.item_info = hyper_paras['items_info']
        self.Z = hyper_paras['Z']
        self.y = hyper_paras['y']
        self.v = hyper_paras['v']
        self.x = hyper_paras['x']
        self.for_cal_sum_by_product = hyper_paras['for_cal_sum_by_product']
        self.Z_Dis_sum = hyper_paras['Z_Dis_sum']
        self.F_Dis_sum = hyper_paras['F_Dis_sum']
        self.sum_input_product = hyper_paras['sum_input_product']
        self.eco_lat_lon = hyper_paras['eco_lat_lon']
        self.num_regions = len(self.regions)
        self.num_commodities = len(self.activities)
        self.num_all_items = self.num_regions * self.num_commodities
        self.sim_for_cal_sum_by_product = self.for_cal_sum_by_product[:self.num_commodities, :]
        self.q_L = self.v / self.x
        self.q_p = np.dot(self.sum_input_product, np.diag(1 / self.x))
        self.q = np.concatenate([self.q_p, np.reshape(self.q_L, (1, self.num_regions * self.num_commodities))], axis=0)
        self.domestic_to_firm_position = np.ones_like(self.Z, dtype=bool) * False
        self.domestic_to_household_position = np.ones_like(self.y, dtype=bool) * False
        for temp in range(self.num_regions):
            self.domestic_to_firm_position[temp * self.num_commodities:((temp + 1) * self.num_commodities), :][:,
            temp * self.num_commodities:((temp + 1) * self.num_commodities)] = True
            self.domestic_to_household_position[temp * self.num_commodities:((temp + 1) * self.num_commodities), temp] \
                = True
        self.total_production_for_each_comm = np.sum(np.reshape(self.x, (self.num_regions, -1)),
                                                     axis=0)
        self.NPI_sector_multiplier_for_each_firm = np.tile(self.NPI_sector_multiplier, self.num_regions)

    def cal_reduction_of_labour(self, epidemic_data, t):
        unavailable_labour_due_to_disease_for_each_firm = np.repeat(epidemic_data['unavailable_l_frac'][t],
                                                                    self.num_commodities)
        reduction_of_contacts_for_each_firm = np.repeat(epidemic_data['reduction_in_contact'][t],
                                                        self.num_commodities)
        return np.maximum(unavailable_labour_due_to_disease_for_each_firm,
                          reduction_of_contacts_for_each_firm * self.NPI_sector_multiplier_for_each_firm)

    def cal_household_demand_for_each_product(self, epidemic_data, t):
        changed_y = self.F_Dis_sum[:self.num_commodities].copy()
        should_change_y_household_index = epidemic_data['I_frac'][t] > self.I_fraction_threshold
        I_over_temp = np.tile(epidemic_data['I_frac'][t] - self.I_fraction_threshold, (self.num_commodities, 1))
        demand_change_result = changed_y * self.demand_multiplier_sector[:, None] * (1 + I_over_temp) / \
                               (self.demand_multiplier_sector[:, None] + I_over_temp)
        changed_y[:, should_change_y_household_index] = demand_change_result[:, should_change_y_household_index]
        return changed_y

    @staticmethod
    def division_define(arr_x, arr_y, default_value):
        return np.divide(arr_x, arr_y, out=np.ones_like(arr_x) * float(default_value), where=arr_y != 0)

    def cal_sum_by_products(self, input_data):
        reshape_input_data = np.reshape(input_data, (self.num_regions, self.num_commodities, -1))
        return np.sum(reshape_input_data, axis=0), reshape_input_data

    def cal_sum_by_regions_for_time_series_arr(self, arr):
        reshape_input_data = np.reshape(arr, (len(arr), self.num_regions, -1))
        return np.sum(reshape_input_data, axis=2)

    def adjust_order_weight(self, original_order_weight, temp_reweight_demand_supply_history):
        adjust_product_input_fraction = original_order_weight * temp_reweight_demand_supply_history[:, None]
        adjust_product_input_fraction_sum, adjust_product_input_fraction_reshape = \
            self.cal_sum_by_products(adjust_product_input_fraction)
        adjust_product_input_fraction = \
            self.division_define(adjust_product_input_fraction_reshape,
                                 adjust_product_input_fraction_sum[None, :, :], 0)
        return adjust_product_input_fraction

    def adjust_kappa(self, current_kappa, current_profit, rho):
        current_x = self.x * current_kappa
        current_world_production_distribution = np.reshape(current_x, (self.num_regions, -1))
        current_world_production_distribution = current_world_production_distribution / \
                                                self.total_production_for_each_comm[None, :]
        weighted_ave_profit = np.sum(current_profit * current_world_production_distribution, axis=0,
                                     keepdims=True)
        var_profit = current_profit - weighted_ave_profit
        change_in_world_production_distribution = \
            current_world_production_distribution * var_profit * rho
        adjust_world_production_distribution = \
            current_world_production_distribution + change_in_world_production_distribution
        kappa = (self.total_production_for_each_comm[None, :] * adjust_world_production_distribution).flatten() / self.x
        return kappa, var_profit.flatten(), adjust_world_production_distribution.flatten()

    def get_last_day_in_epidemic_control_in_eco_time_step(self, NPI_policy_scenario, keep_time_step):
        if NPI_policy_scenario in ['keep_curr', 'linear', 'quad_neg', 'quad_pos']:
            last_day_in_epidemic_control = keep_time_step // self.days_in_each_eco_time_step
            return int(last_day_in_epidemic_control)

    def product_index_list_in_items(self, product_index):
        return [product_index + i * self.num_commodities for i in range(self.num_regions)]

    def cal_production_distribution(self, y_actual_t):
        current_actual_world_production_distribution = np.reshape(y_actual_t, (self.num_regions, -1))
        current_actual_world_production_distribution = \
            current_actual_world_production_distribution / np.sum(current_actual_world_production_distribution, axis=0,
                                                                  keepdims=True)
        return current_actual_world_production_distribution.flatten()

    def IO_trans(self, rho, varphi, NPI_policy_scenario, keep_time_step):
        epidemic_data = \
            {k: np.array(v) for k, v in
             self.epidemic_model.load_epidemic_data_for_economic_cal(NPI_policy_scenario, keep_time_step).items()}
        print('------- successfully load epidemic data for economic cal --------')

        economic_data = {'value_added_actual': [],
                         'value_added_max': [],
                         'value_added_without_evo': [],
                         'production_distribution': [],
                         'actual_production_distribution': [],
                         'profit': [],
                         'reliability': [],
                         'kappa': [],
                         'total_order': [],
                         'var_profit': [],
                         'demand': [],
                         'final_demand': []
                         }

        Inv_t = self.num_time_steps_in_inv_init * self.sum_input_product
        product_to_firms = np.array(self.Z)
        product_to_households = np.array(self.y)
        orderF_t = np.array(self.Z)
        orderH_t = np.array(self.y)
        total_order_t = np.sum(orderF_t, axis=1) + np.sum(orderH_t, axis=1)
        orderF_distribution_fraction_t = orderF_t / total_order_t[:, None]
        orderH_distribution_fraction_t = orderH_t / total_order_t[:, None]
        reliability = np.ones(self.num_all_items)
        kappa = np.ones(self.num_all_items, dtype=float)

        for t in range(self.T + 1):
            economic_data['reliability'].append(reliability)
            economic_data['total_order'].append(total_order_t)


            temporal_received_products_F, _ = self.cal_sum_by_products(product_to_firms)
            temporal_received_products_H, _ = self.cal_sum_by_products(product_to_households)
            reduction_of_labour_t = self.cal_reduction_of_labour(epidemic_data, t)
            labour_input_t = self.v * kappa * (1 - reduction_of_labour_t)
            Inv_t = Inv_t + temporal_received_products_F
            input_matrix_t = np.concatenate([Inv_t, np.reshape(labour_input_t, (1, len(labour_input_t)))], axis=0)
            x_max_t = np.min(self.division_define(input_matrix_t, self.q, np.inf), axis=0)
            x_actual_t = np.minimum(x_max_t, total_order_t)
            labour_profit = total_order_t / (self.x * kappa)  * x_max_t / (self.x * kappa)
            x_actual_t_reshape = x_actual_t[:, None]
            product_to_firms = orderF_distribution_fraction_t * x_actual_t_reshape
            product_to_households = np.multiply(orderH_distribution_fraction_t, x_actual_t_reshape)
            Inv_t = Inv_t - self.q_p * x_actual_t[None, :]
            Inv_t[Inv_t < 0] = 0
            economic_data['value_added_actual'].append(x_actual_t * self.q_L)
            economic_data['value_added_without_evo'].append(labour_input_t / kappa)
            economic_data['value_added_max'].append(labour_input_t)
            reliability = (1 - varphi) * reliability + varphi * self.division_define(x_actual_t, total_order_t, 1)

            profit = np.reshape(labour_profit, (self.num_regions, -1))
            economic_data['profit'].append(profit.flatten())
            kappa, var_profit, adjust_world_production_distribution = self.adjust_kappa(kappa, profit, rho)
            economic_data['var_profit'].append(var_profit)
            economic_data['production_distribution'].append(adjust_world_production_distribution)
            economic_data['kappa'].append(kappa)
            economic_data['actual_production_distribution'].append(self.cal_production_distribution(x_actual_t))
            temp_reweight_demand_supply_history = kappa * reliability
            target_inv_one_time_step = self.x * kappa * (1 - reduction_of_labour_t)
            Target_Inv_t = target_inv_one_time_step[None, :] * self.num_time_steps_in_target_inventory * self.q_p
            DemF_product_t = Target_Inv_t - Inv_t
            DemF_product_t[DemF_product_t < 0] = 0
            economic_data['demand'].append(np.sum(DemF_product_t, axis=0))
            adjust_product_input_fraction_for_F = self.adjust_order_weight(self.Z, temp_reweight_demand_supply_history)

            orderF_t = DemF_product_t[None, :, :] * adjust_product_input_fraction_for_F
            orderF_t = np.reshape(orderF_t, (-1, orderF_t.shape[-1]))
            DemH_product_t = self.cal_household_demand_for_each_product(epidemic_data, t)
            economic_data['final_demand'].append(np.sum(DemH_product_t, axis=0))


            adjust_product_input_fraction_for_H = self.adjust_order_weight(self.y, temp_reweight_demand_supply_history)
            orderH_t = DemH_product_t[None, :, :] * adjust_product_input_fraction_for_H
            orderH_t = np.reshape(orderH_t, (-1, orderH_t.shape[-1]))
            total_order_t = np.sum(orderH_t, axis=1) + np.sum(orderF_t, axis=1)
            total_order_t_reshape = total_order_t[:, None]

            if sum(total_order_t != 0) == len(total_order_t):
                orderF_distribution_fraction_t = np.divide(orderF_t, total_order_t_reshape)
                orderH_distribution_fraction_t = np.divide(orderH_t, total_order_t_reshape)
            else:
                orderF_distribution_fraction_t = self.division_define(orderF_t, total_order_t_reshape, 0)
                orderH_distribution_fraction_t = self.division_define(orderH_t, total_order_t_reshape, 0)

        economic_data['value_added_original'] = self.v
        economic_data['epidemic_data'] = epidemic_data

        return economic_data

    @staticmethod
    def get_eco_region_CHI_class(economic_region_npi_policy, country_index_in_eco_model):
        for CHI_class in ['mild_npi', 'moderate_npi', 'stringent_npi']:
            if country_index_in_eco_model in economic_region_npi_policy[CHI_class]:
                return CHI_class

    def load_economic_data(self, rho, varphi, NPI_policy_scenario, keep_time_step, if_check_economic_data=False):
        if if_check_economic_data:
            self.epidemic_model.load_epidemic_data_for_economic_cal(NPI_policy_scenario, keep_time_step)
        scenario_key = \
            str(rho) + '_' + str(varphi) + '_' + \
            NPI_policy_scenario + '_' + str(keep_time_step / (52 * 7))
        save_path = self.hyper_file_path + '/eco_data_' + scenario_key + '.pkl'
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                eco_epi_data = pickle.load(f)
        else:
            eco_epi_data = self.IO_trans(rho, varphi, NPI_policy_scenario, keep_time_step)
            with open(save_path, 'wb') as f:
                pickle.dump(eco_epi_data, f)
        return scenario_key, eco_epi_data
