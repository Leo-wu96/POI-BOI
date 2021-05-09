% clear;
% gender = load('../data/doc2vec_log/gender_doc2vec_result.mat');
% age = load('../data/doc2vec_log/age_doc2vec_result.mat');
% gender = load('../data/bert_log/gender_bert_result.mat');
% age = load('../data/bert_log/age_bert_result.mat');
% data = load('../data/model_export_Acc.npy.mat');
s_score_d2v = load('../data/doc2vec_log/elbow.mat');
s_score_bert = load('../data/bert_log/elbow.mat');
% 
% figure(1);
% subplot(121);
% plot(gender.ad,'*-','LineWidth',2);
% hold on
% plot(gender.adver,'*-','LineWidth',2);
% hold on
% plot(gender.creative,'*-','LineWidth',2);
% hold on
% plot(gender.feature3,'*-','LineWidth',2);
% grid on
% xlabel('Epoch');
% ylabel('Accuracy');
% title('Gender');
% legend({'semantic POI','user''s behavior','physical location','3 attributes fusion'});
% set(gcf, 'Color', [1,1,1]);
% 
% subplot(122);
% plot(age.ad,'*-','LineWidth',2);
% hold on
% plot(age.adver,'*-','LineWidth',2);
% hold on
% plot(age.creative,'*-','LineWidth',2);
% hold on
% plot(age.feature3,'*-','LineWidth',2);
% grid on
% xlabel('Epoch');
% ylabel('Accuracy');
% title('Age');
% legend({'semantic POI','user''s behavior','physical location','3 attributes fusion'});
% set(gcf, 'Color', [1,1,1]);


figure(2);
subplot(211);
plot(s_score_d2v.elbow,'*-','LineWidth',2);
set(gca,'xtick',[1:2:99],'xticklabel',{2:4:98});
xlabel('Number of clusters');
ylabel('Distortion');
title('Elbow for Doc2Vec');
set(gcf, 'Color', [1,1,1]);
grid on;
subplot(212);
plot(s_score_bert.elbow,'*-','LineWidth',2);
set(gca,'xtick',[1:2:99],'xticklabel',{2:4:98});
xlabel('Number of clusters');
ylabel('Distortion');
title('Elbow for BERT');
grid on;
set(gcf, 'Color', [1,1,1]);


% figure(3);
% subplot(121);
% plot(data.adg,'*-','LineWidth',2);
% hold on
% plot(data.adverg,'*-','LineWidth',2);
% hold on
% plot(data.cg,'*-','LineWidth',2);
% hold on
% plot(data.fg,'*-','LineWidth',2);
% grid on
% xlabel('Epoch');
% ylabel('Accuracy');
% title('Gender');
% legend({'semantic POI','user''s behavior','physical location','3 attributes fusion'});
% set(gcf, 'Color', [1,1,1]);
% 
% subplot(122);
% plot(data.ada,'*-','LineWidth',2);
% hold on
% plot(data.advera,'*-','LineWidth',2);
% hold on
% plot(data.ca,'*-','LineWidth',2);
% hold on
% plot(data.fa,'*-','LineWidth',2);
% grid on
% xlabel('Epoch');
% ylabel('Accuracy');
% title('Age');
% legend({'semantic POI','user''s behavior','physical location','3 attributes fusion'});
% set(gcf, 'Color', [1,1,1]);


% figure(4);
% subplot(131);
% plot(age1.ad,'LineWidth',2);
% hold on
% plot(age1.adver,'LineWidth',2);
% hold on
% plot(age1.creative,'LineWidth',2);
% hold on
% plot(age1.feature3,'LineWidth',2);
% grid on
% xlabel('Epoch');
% ylabel('Accuracy');
% legend({'semantic POI','user''s behavior','physical location','3 attributes fusion'});
% set(gcf, 'Color', [1,1,1]);
% 
% subplot(132);
% plot(age2.ad,'LineWidth',2);
% hold on
% plot(age2.adver,'LineWidth',2);
% hold on
% plot(age2.creative,'LineWidth',2);
% hold on
% plot(age2.feature3,'LineWidth',2);
% grid on
% xlabel('Epoch');
% ylabel('Accuracy');
% legend({'semantic POI','user''s behavior','physical location','3 attributes fusion'});
% set(gcf, 'Color', [1,1,1]);
% 
% subplot(133);
% plot(data.ada,'LineWidth',2);
% hold on
% plot(data.advera,'LineWidth',2);
% hold on
% plot(data.ca,'LineWidth',2);
% hold on
% plot(data.fa,'LineWidth',2);
% grid on
% xlabel('Epoch');
% ylabel('Accuracy');
% legend({'semantic POI','user''s behavior','physical location','3 attributes fusion'});
% set(gcf, 'Color', [1,1,1]);



% compare = load('../data/model_export_Acc.npy_compare_module.mat');
% figure(9);
% subplot(121);
% plot(compare.fg,'*-','LineWidth',2);
% hold on
% plot(compare.cg,'*-','LineWidth',2);
% grid on
% xlabel('Epoch');
% ylabel('Accuracy');
% title('Gender');
% legend({'Add Merge layer','No Merge layer'});
% set(gcf, 'Color', [1,1,1]);
% 
% subplot(122);
% plot(compare.fa,'*-','LineWidth',2);
% hold on
% plot(compare.c,'*-','LineWidth',2);
% grid on
% xlabel('Epoch');
% ylabel('Accuracy');
% title('Age');
% legend({'Add Merge layer','No Merge layer'});
% set(gcf, 'Color', [1,1,1]);
