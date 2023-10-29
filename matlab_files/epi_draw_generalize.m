function epi_draw_generalize(raw_epidemic_data_info, I_thre, Re_thre, phi, k, CHI_thre, rho, ...
    varphi, ylim1, ylim2,yticks2, if_base, supple_name, line_info, line_info_NPI)
main_file_name = strcat('main_rho_', rho, '_varphi_', varphi);

eco_epi_hyper_paras_info = strcat( 'I_thre_', I_thre,'_Re_thre_',Re_thre,'_phi_',phi,'_k_',k,'_CHI_thre_',CHI_thre);
% path
results_path = strcat('results/', raw_epidemic_data_info, '/', eco_epi_hyper_paras_info, '/', main_file_name, '/main_results.csv');

save_figs_dir = strcat('results/', raw_epidemic_data_info, '/', eco_epi_hyper_paras_info,'/', main_file_name, '/main_plot');
mkdir(save_figs_dir)

save_supple_figs_dir = strcat('results/', raw_epidemic_data_info,'/supple_figs_summary');
mkdir(save_supple_figs_dir)

if if_base
    figure_save_path = strcat('results/', raw_epidemic_data_info, '/', eco_epi_hyper_paras_info,'/', main_file_name, '/main_plot/epi_results.eps');
else
    figure_save_path = strcat('results/', raw_epidemic_data_info, '/supple_figs_summary/', supple_name,'epi_results.eps');
end

%% ---------- read results -----------------------
results = readtable(results_path,'PreserveVariableNames',true);

%% --------- plot settings --------------------------
row_info = {'_prev_', '_cum_mor_'};
titles = {'World', 'High-CHI countries/regions', 'Mid-CHI countries/regions', 'Low-CHI countries/regions'};
col_info = {'world', 'stringent_npi', 'moderate_npi',  'mild_npi'};
point_length = 1; 
texts = char(97:120);
point_interval = 2;
marker_size = 2;
line_width = 2;
font_size = 15;
         
 color_all = [[123/255,204/255,196/255;
             43/255 140/255 190/255;
             3/255,78/255,123/255;];
             
             [252/255 146/255 124/255;
             239/255 59/255 44/255;
             153/255 0/255 13/255;];
             
             [254/255,153/255,41/255;
             255/255,127/255,0/255;
             204/255,76/255,2/255;];
             
             [102/255,194/255,164/255;
             35/255,139/255,69/255;
            0/255,88/255,36/255;]];
            
line_style = {':', '-.', '-'};   
figure('Position', [50,344,975,446])

%% ------- plot ----------------------------------
one_year_x = 1 + 52/point_length;

figs = 1;
for row=1:2
    for col=1:4
        subplot(2,4,4*(row-1)+col,'Position', [0.063+(col-1)*0.24, 0.59-(row-1)*0.48, 0.2, 0.35],'Units','normalized')
        h_all = [];
        
        for i=1:3
            base_scenario_key = strcat(rho, '_', varphi, '_', line_info_NPI(i)); 
            line_ = string(strcat(base_scenario_key, line_info(i), row_info(row), col_info(col)));
            line_result = results.(line_)*100;
            % color_all((col-1)*3+i,:)
            if if_base
                h = plot(line_result(52*1/point_length:point_length:length(line_result)),string(line_style(i)), 'LineWidth', line_width, 'color', color_all((col-1)*3+i,:));
            else
               h = plot(line_result(1:point_length:length(line_result)),string(line_style(i)), 'LineWidth', line_width, 'color', color_all((col-1)*3+i,:)); 
            end
               h_all = [h_all, h];
            hold on
           
        end
        
        if row==1
        legend(h_all,{'Explosive reopening', 'Steady reopening','Incremental reopening'}, ...
            'Position', [0.1 + (col-1) * 0.24, 0.75 - (row-1) * 0.6,0.155,0.156],'FontSize',font_size-2.5)

        legend boxoff 
        end
        
        set(gca,'FontSize',font_size)
        box off
        %xlim([0 (length(line_result)-1-52)/point_length])
        
        if row==1
             ylim(ylim1) % [0 10]
        else
            ylim(ylim2) % [0 1.2]
            yticks(yticks2) % [0 0.6 1.2]
        end
        if col>1
            yticks([])
        end
        if if_base
            xlim([0 (length(line_result)-1-52)/point_length])
            xticks(0:52*1/point_length:(length(line_result)-1-52)/point_length)
            xticklabels({'1','2','3','4','5'})
        else
            xlim([0 (length(line_result)-1)/point_length])
            xticks(0:52*1/point_length:(length(line_result)-1)/point_length)
            xticklabels({'0', '1','2','3','4','5'})
        end
        
        
        
        if col==1
            if row == 1
                    ylabel('Prevalence (%)','FontSize',font_size, 'Units', 'normalized' ,'Position', [-0.194,0.5,0])
            else if row==2
                    ylabel({ 'Cumulative mortality rate (%)'},'FontSize',font_size, 'Units', 'normalized' ,'Position', [-0.194,0.49,0])
                end
            end
        end
        
       if row==1
            title(titles(col),'FontSize',font_size,'FontWeight','normal')
        end
        
        if row==2
            xlabel('t [years]','FontSize',font_size)
        end
        

        if if_base
            text(-0.15, 1.1, texts(figs+1), 'Units', 'Normalized','FontSize',font_size,'FontWeight','bold');
            disp(figs)
        else
            text(-0.15, 1.1, texts(figs), 'Units', 'Normalized','FontSize',font_size,'FontWeight','bold');
            disp(figs)
        end
        figs = figs +1;
        
        
    end
end
saveas(gcf,figure_save_path,'epsc')
end
