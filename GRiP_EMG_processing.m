% With this script, we are breaking up the larger file of raw EMG data into
% portions that are more relevant to our project
% The structure is divided into sections based on the exercise, restimulus,
% rerepetition, and the EMG signals that correspond to them
%%

ProcessedEMG = [];
for gg = 1:27  % 27 subjects
    for hh = 1:3 % 3 exercises per subject
        clearvars -except gg hh ProcessedEMG % clearing data from previous exercises
        foldername = ['s' num2str(gg)]; % dynamically naming folders
        filename = ['S' num2str(gg) '_A1_E' num2str(hh)]; % dynamically naming files
        load(fullfile(foldername,filename)); % dynamically loading files
        for ii = 1:max(restimulus) % defining quantity of restimulus fields
            fname3 = ['Restimulus_' num2str(ii)]; % naming restimulus fields
            for jj = 1:max(rerepetition) % defining quantity of rerep fields
                fname4 = ['Rerepetition_' num2str(jj)]; % naming rerep fields
                RestimIndex = find(restimulus == ii); 
                RerepIndex = find(rerepetition == jj);
                ProcessedEMG.(foldername).(filename).(fname3).(fname4) = emg(intersect(RestimIndex,RerepIndex),:); % choosing corresponding data points
            end
        end
    end
end


save DB1processed.mat ProcessedEMG


