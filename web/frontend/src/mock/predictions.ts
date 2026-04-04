import type { PredictionResponse } from '../api/queries'

// ---------------------------------------------------------------------------
// Query 1: BAV1341.13v.1 — canon on marriage condemnation
// ---------------------------------------------------------------------------

const PREDICTIONS_1: PredictionResponse = {
  file_id: 1,
  filename: 'BAV1341.13v.1.txt',
  model: 'bowphs_LaTa',
  predictions: [
    {
      rank: 1,
      dir_name: 'Can.apost.42',
      score: 0.591,
      dir_files: ['Can.apost.42_a.txt', 'Can.apost.42_b.txt'],
      preview_text: 'XLII. Si quis episcopus aut presbiter aut diaconus alea',
      candidate_files: [
        {
          filename: 'Can.apost.42_a.txt',
          text: 'XLII. Si quis episcopus aut presbiter aut diaconus aleae id est tabulae ludo se voluptuosius dederit, aut desinat aut certe damnetur.',
        },
        {
          filename: 'Can.apost.42_b.txt',
          text: 'Si quis episcopus vel presbiter vel diaconus aleae id est tabulae ludo se dediderit, aut desinat aut deponatur. Si subdiaconus aut lector aut cantor similia fecerit, aut desinat aut communione privetur.',
        },
      ],
    },
    {
      rank: 2,
      dir_name: 'Can.apost.44',
      score: 0.563,
      dir_files: ['Can.apost.44_a.txt'],
      preview_text: 'XLIIII. Episcopus aut presbiter aut diaconus usuras a d',
      candidate_files: [
        {
          filename: 'Can.apost.44_a.txt',
          text: 'XLIIII. Episcopus aut presbiter aut diaconus usuras a debitoribus exigens aut desinat aut deponatur.',
        },
      ],
    },
    {
      rank: 3,
      dir_name: 'CNEO.315.6',
      score: 0.482,
      dir_files: ['CNEO.315.6_a.txt', 'CNEO.315.6_b.txt', 'CNEO.315.6_c.txt'],
      preview_text: 'Si quis dixerit nuptias detestabiles esse et procreatio',
      candidate_files: [
        {
          filename: 'CNEO.315.6_a.txt',
          text: 'Si quis dixerit nuptias detestabiles esse et procreationem filiorum malam et perniciosam, et sicut Manichaeus et Priscillianus dixerunt, anathema sit.',
        },
        {
          filename: 'CNEO.315.6_b.txt',
          text: 'Item placuit ut si quis nuptias dampnaverit et dormientem cum viro fidelem et religiosam mulierem abhominandam putaverit, anathema sit.',
        },
        {
          filename: 'CNEO.315.6_c.txt',
          text: 'Quicumque dixerit matrimonium esse malum et a diabolo institutum, sicut Manichaeus docuit, anathema sit in perpetuum.',
        },
      ],
    },
    {
      rank: 4,
      dir_name: 'Ambrose, De spirit.sanct.3.6 (ed. Faller, 164-65, lines 12-22)',
      score: 0.341,
      dir_files: ['Ambrose_DeSpirit_3.6.txt'],
      preview_text: 'Et ideo non corporali sed spirituali consortio coniuges',
      candidate_files: [
        {
          filename: 'Ambrose_DeSpirit_3.6.txt',
          text: 'Et ideo non corporali sed spirituali consortio coniuges sumus Christi, quia unum sumus cum eo, et ipse nos redemit sanguine suo. Unde apostolus ait: Viri diligite uxores vestras sicut et Christus dilexit ecclesiam.',
        },
      ],
    },
    {
      rank: 5,
      dir_name: 'Source unidentified',
      score: 0.298,
      dir_files: ['unident_frag_001.txt'],
      preview_text: 'Quod si quis autem nuptias in accusationem deduxerit et',
      candidate_files: [
        {
          filename: 'unident_frag_001.txt',
          text: 'Quod si quis autem nuptias in accusationem deduxerit et mulierem fidelem abhominandam crediderit, aut etiam accusandam tamquam quae non possit regnum dei ingredi, anathema sit.',
        },
      ],
    },
  ],
}

// ---------------------------------------------------------------------------
// Query 2: BAV1341.14r.2 — abstinence from meats
// ---------------------------------------------------------------------------

const PREDICTIONS_2: PredictionResponse = {
  file_id: 2,
  filename: 'BAV1341.14r.2.txt',
  model: 'bowphs_LaTa',
  predictions: [
    {
      rank: 1,
      dir_name: 'Can.apost.51',
      score: 0.624,
      dir_files: ['Can.apost.51_a.txt', 'Can.apost.51_b.txt'],
      preview_text: 'Si quis episcopus aut presbiter aut diaconus aut omnino',
      candidate_files: [
        {
          filename: 'Can.apost.51_a.txt',
          text: 'Si quis episcopus aut presbiter aut diaconus aut omnino ex catalogo clericorum a nuptiis et carnibus et vino non propter exercitationem sed propter abominationem abstineat, oblitus quod omnia valde bona sunt.',
        },
        {
          filename: 'Can.apost.51_b.txt',
          text: 'LI. Si quis episcopus vel presbiter vel diaconus vel omnino de sacerdotali catalogo a nuptiis et carnibus et vino non pro exercitio abstinentiae sed pro execratione se abstinuerit, aut corrigatur aut deponatur.',
        },
      ],
    },
    {
      rank: 2,
      dir_name: 'Gangr.c.2',
      score: 0.578,
      dir_files: ['Gangr.c.2_a.txt'],
      preview_text: 'Si quis carnes edentes damnat et non propter exercitium',
      candidate_files: [
        {
          filename: 'Gangr.c.2_a.txt',
          text: 'Si quis carnes edentes damnat et non propter exercitium et continentiam sed tamquam abominabiles eas detestatur, oblitus eorum quae propter usum hominum data sunt, anathema sit.',
        },
      ],
    },
    {
      rank: 3,
      dir_name: 'Gangr.c.14',
      score: 0.501,
      dir_files: ['Gangr.c.14_a.txt'],
      preview_text: 'Si qua mulier relinquat virum suum et recedere velit nu',
      candidate_files: [
        {
          filename: 'Gangr.c.14_a.txt',
          text: 'Si qua mulier relinquat virum suum et recedere velit nuptias abominando, anathema sit.',
        },
      ],
    },
    {
      rank: 4,
      dir_name: 'CNEO.315.2',
      score: 0.423,
      dir_files: ['CNEO.315.2_a.txt', 'CNEO.315.2_b.txt'],
      preview_text: 'Si quis presbiter aut diaconus a carnibus se abstinueri',
      candidate_files: [
        {
          filename: 'CNEO.315.2_a.txt',
          text: 'Si quis presbiter aut diaconus a carnibus se abstinuerit non propter exercitationem sed propter execrationem, quia creaturam dei existimat immundam, aut corrigatur aut deponatur.',
        },
        {
          filename: 'CNEO.315.2_b.txt',
          text: 'Placuit ut clericus qui a carnibus et vino abstinet non pro exercitatione virtutis sed pro detestatione creaturae, aut corrigatur aut de ecclesia proiciatur.',
        },
      ],
    },
    {
      rank: 5,
      dir_name: 'Jerome, Adv.Jovin.2.7',
      score: 0.312,
      dir_files: ['Jerome_AdvJovin_2.7.txt'],
      preview_text: 'Non quod nuptias condemnemus sed quod virginitatem prae',
      candidate_files: [
        {
          filename: 'Jerome_AdvJovin_2.7.txt',
          text: 'Non quod nuptias condemnemus sed quod virginitatem praeferamus. Non quod abstinentiam a cibis praecipimus sed quod temperantiam laudamus. Aliud est enim damnare creaturam, aliud moderari usum eius.',
        },
      ],
    },
  ],
}

// ---------------------------------------------------------------------------
// Query 3: Hat42.148v.3 — excommunication without cause
// ---------------------------------------------------------------------------

const PREDICTIONS_3: PredictionResponse = {
  file_id: 3,
  filename: 'Hat42.148v.3.txt',
  model: 'bowphs_LaTa',
  predictions: [
    {
      rank: 1,
      dir_name: 'Can.apost.41',
      score: 0.647,
      dir_files: ['Can.apost.41_a.txt', 'Can.apost.41_b.txt'],
      preview_text: 'Praecipimus episcopum habere rerum ecclesiasticarum pot',
      candidate_files: [
        {
          filename: 'Can.apost.41_a.txt',
          text: 'Praecipimus episcopum habere rerum ecclesiasticarum potestatem. Si enim animae hominum pretiosae illi sunt commissae, multo magis eum de pecuniis oportet mandare.',
        },
        {
          filename: 'Can.apost.41_b.txt',
          text: 'XLI. Episcopus habeat ecclesiasticarum rerum curam et eas dispenset tamquam deo contemplante. Non ei liceat autem aliquid ex eis suis propinquis largiri.',
        },
      ],
    },
    {
      rank: 2,
      dir_name: 'Can.apost.36',
      score: 0.589,
      dir_files: ['Can.apost.36_a.txt'],
      preview_text: 'Si quis episcopus nominatus et non susceperit ministeriu',
      candidate_files: [
        {
          filename: 'Can.apost.36_a.txt',
          text: 'Si quis episcopus nominatus et non susceperit ministerium et curam populi sibi commissam, hic sit excommunicatus donec suscipiat. Similiter presbiter et diaconus.',
        },
      ],
    },
    {
      rank: 3,
      dir_name: 'Nic.325.c.5',
      score: 0.534,
      dir_files: ['Nic.325.c.5_a.txt', 'Nic.325.c.5_b.txt'],
      preview_text: 'De his qui communione privantur, sive ex clero sive ex',
      candidate_files: [
        {
          filename: 'Nic.325.c.5_a.txt',
          text: 'De his qui communione privantur, sive ex clero sive ex laico ordine, ab episcopis per unamquamque provinciam sententia regularis obtineat, ut ii qui abiiciuntur ab aliis non recipiantur.',
        },
        {
          filename: 'Nic.325.c.5_b.txt',
          text: 'Requiratur autem ne pusillanimitate aut contentione vel alio quolibet episcopi vitio videantur a congregatione seclusu. Ut hoc ergo decentius inquiratur, bene placuit annis singulis per unamquamque provinciam bis in anno concilia celebrari.',
        },
      ],
    },
    {
      rank: 4,
      dir_name: 'Antioch.c.6',
      score: 0.401,
      dir_files: ['Antioch.c.6_a.txt'],
      preview_text: 'Si quis a proprio episcopo excommunicatus est, non priu',
      candidate_files: [
        {
          filename: 'Antioch.c.6_a.txt',
          text: 'Si quis a proprio episcopo excommunicatus est, non prius ab aliis suscipiatur quam a suo fuerit episcopo receptus, aut certe concilio facto respondeat et episcopum convincat atque ita recipiatur.',
        },
      ],
    },
    {
      rank: 5,
      dir_name: 'Sardic.c.13',
      score: 0.356,
      dir_files: ['Sardic.c.13_a.txt'],
      preview_text: 'Osius episcopus dixit: Illud quoque statuendum est ut n',
      candidate_files: [
        {
          filename: 'Sardic.c.13_a.txt',
          text: 'Osius episcopus dixit: Illud quoque statuendum est ut nullus episcopus alterius episcopi civitatis clericum aut laicum excommunicatum in communionem suscipiat, ne episcoporum contemptus ac disciplina vilescat.',
        },
      ],
    },
  ],
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

export const MOCK_PREDICTIONS: Map<number, PredictionResponse> = new Map([
  [1, PREDICTIONS_1],
  [2, PREDICTIONS_2],
  [3, PREDICTIONS_3],
])
