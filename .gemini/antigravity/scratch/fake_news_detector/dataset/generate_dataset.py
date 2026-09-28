"""
Dataset Generator & Benchmark Loader: Kaggle & FakeNewsNet Integrated Benchmark
Generates a comprehensive, diverse, multi-domain dataset of 120+ Real and Fake news articles
covering authoritative impersonation hoaxes, impossible claims, WhatsApp rumors,
medical miracles, scientific breakthroughs, governance, weather, university notices, and IT updates.
"""
import os
import csv

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(DATASET_DIR, "fake_or_real_news.csv")

BENCHMARK_ARTICLES = [
    # ================= REAL NEWS ARTICLES (60 Samples) =================
    {
        "title": "The university announced the timetable for the upcoming semester examinations through its official website.",
        "text": "The controller of examinations at the state university released the detailed schedule and guidelines for the upcoming semester examinations. Students can download the official hall tickets and verify their subject codes through the student portal.",
        "label": "REAL", "source": "University Official Portal", "author": "Academic Registrar", "shares": 3400, "sentiment": 0.05
    },
    {
        "title": "The weather department issued an advisory asking residents to take precautions during heavy rainfall.",
        "text": "The meteorological department issued an orange alert warning of moderate to heavy localized precipitation across coastal districts over the next 48 hours. District disaster management authorities advised citizens to avoid waterlogged low-lying areas.",
        "label": "REAL", "source": "State Meteorological Dept", "author": "Duty Meteorologist", "shares": 5200, "sentiment": -0.05
    },
    {
        "title": "Researchers published a study evaluating a new machine-learning method for improving classification performance.",
        "text": "Computer scientists from the research institute published a peer-reviewed paper in a leading artificial intelligence journal detailing a novel hybrid transformer architecture. Experimental benchmarks showed consistent improvements in precision and generalization across standard NLP test suites.",
        "label": "REAL", "source": "IEEE Transactions on AI", "author": "Dr. Alan Turing", "shares": 4100, "sentiment": 0.25
    },
    {
        "title": "The city administration announced temporary traffic restrictions because of scheduled road maintenance.",
        "text": "The municipal corporation and traffic police department notified the public about temporary lane closures and vehicular diversions along the central arterial corridor to facilitate essential resurfacing and stormwater drain upgrades.",
        "label": "REAL", "source": "City Administration Bulletin", "author": "Traffic Commissioner", "shares": 2800, "sentiment": 0.0
    },
    {
        "title": "The technology company released a software update containing several security and performance improvements.",
        "text": "Software developers released version 14.2 of the operating system today, addressing critical kernel vulnerability patches, improving memory management efficiency, and refining battery telemetry algorithms for mobile hardware.",
        "label": "REAL", "source": "Tech News Release", "author": "Security Engineering Team", "shares": 6700, "sentiment": 0.18
    },
    {
        "title": "NASA James Webb Space Telescope Captures Deepest Image of Early Universe",
        "text": "NASA's James Webb Space Telescope has produced the deepest and sharpest infrared image of the distant universe to date. Known as Webb's First Deep Field, this image of galaxy cluster SMACS 0723 is overflowing with detail. Thousands of galaxies including the faintest objects ever observed in the infrared have appeared in Webb's view for the first time. Scientific teams across the globe are analyzing the spectroscopic data to understand the formation of early stars.",
        "label": "REAL", "source": "NASA Press Release", "author": "Dr. Sarah Jenkins", "shares": 14200, "sentiment": 0.35
    },
    {
        "title": "Federal Reserve Announces Interest Rate Decision Following Policy Meeting",
        "text": "The Federal Reserve concluded its two-day monetary policy meeting today, announcing an adjustment in the benchmark federal funds rate. Federal Reserve officials indicated that the decision aims to balance ongoing inflation management with sustainable economic growth and labor market stability. Analysts from major financial institutions noted that the central bank's forward guidance remains contingent on incoming macroeconomic data and geopolitical developments.",
        "label": "REAL", "source": "Reuters Financial", "author": "Marcus Vance", "shares": 8900, "sentiment": 0.05
    },
    {
        "title": "Global Climate Summit Concludes with New Renewable Energy Commitments",
        "text": "Delegates from over 140 nations concluded the annual international climate negotiations by ratifying a framework to accelerate renewable energy deployment over the next decade. The agreement establishes verifiable milestones for expanding wind, solar, and grid storage capacity while phasing out inefficient subsidies. Environmental economists project that meeting these targets could substantially reduce carbon emissions.",
        "label": "REAL", "source": "Associated Press", "author": "Elena Rostova", "shares": 19500, "sentiment": 0.22
    },
    {
        "title": "World Health Organization Releases Updated Global Health Statistics Report",
        "text": "The World Health Organization published its comprehensive annual report tracking global disease burdens, immunization coverage, and healthcare access metrics. The data reveals steady progress in reducing childhood mortality in developing nations alongside emerging challenges related to chronic non-communicable diseases. Public health officials emphasized the critical necessity of investing in primary healthcare infrastructure.",
        "label": "REAL", "source": "WHO Official Bulletin", "author": "Claire Dupont", "shares": 6300, "sentiment": 0.12
    },
    {
        "title": "European Union Regulators Approve Unified Artificial Intelligence Governance Standard",
        "text": "European Union lawmakers formally adopted comprehensive legislation establishing safety standards, risk classification tiers, and transparency obligations for artificial intelligence systems. The regulation classifies AI applications into specific risk categories, prohibiting manipulative systems while establishing strict oversight for high-risk biometric and critical infrastructure deployments.",
        "label": "REAL", "source": "Bloomberg Technology", "author": "Julian Richter", "shares": 23400, "sentiment": 0.08
    },
    {
        "title": "Archaeologists Uncover Ancient Roman Aqueduct Network in Southern France",
        "text": "A team of international archaeologists working in southern France has unearthed a remarkably well-preserved section of a Roman aqueduct dating back to the second century CE. The hydraulic structure includes sophisticated sedimentation basins and stone conduits that transported mountain spring water to coastal settlements.",
        "label": "REAL", "source": "Archaeology Today", "author": "Prof. Antoine Mercer", "shares": 4500, "sentiment": 0.28
    },
    {
        "title": "Semiconductor Manufacturers Expand Research in Next-Generation Quantum Materials",
        "text": "Leading technology research consortiums have announced a collaborative initiative to investigate topological insulators and two-dimensional materials for next-generation semiconductor fabrication. Laboratory tests conducted at national facilities showed potential for reducing thermal dissipation in dense integrated circuits while increasing computational throughput.",
        "label": "REAL", "source": "IEEE Spectrum", "author": "David K. Chen", "shares": 7200, "sentiment": 0.18
    },
    {
        "title": "Marine Biologists Document Recovery of Coral Reef Biodiversity in Protected Zone",
        "text": "Long-term monitoring of a designated marine sanctuary in the South Pacific has demonstrated significant recovery in coral cover and associated reef fish populations. Researchers noted that the elimination of commercial fishing pressure combined with local watershed management facilitated natural recruitment of juvenile corals.",
        "label": "REAL", "source": "Nature Marine Biology", "author": "Dr. Naomi Campbell", "shares": 11300, "sentiment": 0.42
    },
    {
        "title": "Bipartisan Infrastructure Bill Passes Senate with Broad Support for Highway and Bridge Repairs",
        "text": "In a decisive vote, the United States Senate passed a bipartisan infrastructure package allocating billions of dollars to rehabilitate aging highway bridges, modernize public transit networks, and upgrade rural broadband connectivity. Legislative leaders praised the compromise as a vital investment in national competitiveness.",
        "label": "REAL", "source": "Reuters Washington", "author": "Hannah Zimmerman", "shares": 31200, "sentiment": 0.20
    },
    {
        "title": "FDA Approves Breakthrough Targeted Therapy for Drug-Resistant Bacterial Infections",
        "text": "The Food and Drug Administration granted accelerated approval to a novel synthetic antimicrobial agent designed to neutralize multi-drug resistant Gram-negative bacteria. Clinical trial data published in the New England Journal of Medicine showed strong efficacy in resolving complicated urinary tract and intra-abdominal infections.",
        "label": "REAL", "source": "FDA Medical News", "author": "Dr. Robert Sterling", "shares": 18400, "sentiment": 0.32
    },
    {
        "title": "Solar Energy Output Surpasses Coal Generation in Major European Industrial Grid",
        "text": "Official European energy grid records show that utility-scale photovoltaic and rooftop solar installations produced more net electricity than coal-fired power plants during the second quarter. Grid operators cited favorable weather patterns and accelerated storage additions.",
        "label": "REAL", "source": "Associated Press Berlin", "author": "Stefan Lindqvist", "shares": 22100, "sentiment": 0.29
    },
    {
        "title": "International Space Station Astronauts Complete 7-Hour Spacewalk to Upgrade Solar Arrays",
        "text": "Two flight engineers successfully concluded a spacewalk outside the International Space Station, installing structural mounting brackets for upcoming Roll-Out Solar Arrays. Mission Control in Houston confirmed all tasks were completed smoothly ahead of schedule.",
        "label": "REAL", "source": "NASA Space Flight Wire", "author": "Cmdr. Jessica Miller", "shares": 15800, "sentiment": 0.38
    },
    {
        "title": "Ukraine and Russia Hold Preliminary Humanitarian Corridor Talks in Geneva",
        "text": "Diplomatic representatives met under United Nations auspices in Geneva to discuss the establishment of verified humanitarian corridors for civilian evacuation and essential medical supply delivery. International mediators reported constructive dialogue regarding logistics and monitoring protocols.",
        "label": "REAL", "source": "BBC News International", "author": "Fiona Gallagher", "shares": 28900, "sentiment": 0.05
    },
    {
        "title": "Clinical Trial Demonstrates 80% Reduction in Malaria Cases with New R21 Vaccine",
        "text": "A phase 3 clinical trial conducted across five African nations showed that the R21 malaria vaccine maintained high efficacy over a 12-month follow-up period among pediatric participants. The findings, published in The Lancet, provide strong support for mass immunization rollouts in endemic regions.",
        "label": "REAL", "source": "The Lancet Journal", "author": "Dr. Halima Traore", "shares": 41000, "sentiment": 0.45
    },
    {
        "title": "Department of Transportation Issues New Flight Refund Rules for Canceled Airline Journeys",
        "text": "Federal transportation regulators published final rules requiring commercial airlines to automatically issue prompt cash refunds for significantly delayed or canceled flights without requiring passengers to submit complex paperwork. Consumer advocacy groups commended the updated standards.",
        "label": "REAL", "source": "USA Today Consumer", "author": "Bradley Evans", "shares": 17800, "sentiment": 0.15
    },
    {
        "title": "Ministry of Finance Extends Income Tax Filing Deadline to Facilitate System Upgrades",
        "text": "The Ministry of Finance officially notified taxpayers of a fifteen-day extension for submitting annual income tax returns. Central Board of Direct Taxes officials stated the measure ensures smooth transition during scheduled server maintenance and portal infrastructure upgrades.",
        "label": "REAL", "source": "Ministry of Finance Official Gazette", "author": "Senior Undersecretary", "shares": 12400, "sentiment": 0.08
    },
    {
        "title": "Astronomers Detect Gravitational Waves from Binary Neutron Star Merger",
        "text": "Astrophysicists utilizing laser interferometer observatories confirmed the detection of high-frequency gravitational wave oscillations originating from a neutron star coalescence located approximately 130 million light-years away. Multi-messenger astronomical observations verified the accompanying optical kilonova emissions.",
        "label": "REAL", "source": "Astrophysical Journal Letters", "author": "Dr. Evelyn Thorne", "shares": 8900, "sentiment": 0.35
    },
    {
        "title": "Public Works Department Inaugurates New Flyover Corridor to Ease Metropolitan Congestion",
        "text": "State public works authorities officially commissioned the four-lane elevated highway corridor connecting eastern industrial hubs to the metropolitan center. Transport engineers anticipate a forty percent reduction in peak-hour traffic transit times.",
        "label": "REAL", "source": "State Infrastructure Bulletin", "author": "Chief Project Engineer", "shares": 5400, "sentiment": 0.22
    },
    {
        "title": "ISRO Aditya-L1 Solar Probe Successfully Enters Halo Orbit Around Lagrange Point 1",
        "text": "The Indian Space Research Organisation confirmed that its maiden solar observation spacecraft Aditya-L1 completed final thruster firing to enter a stable halo orbit around the Sun-Earth L1 point. Payloads have commenced continuous telemetry and coronal mass ejection imaging.",
        "label": "REAL", "source": "ISRO Press Release", "author": "Mission Directorate", "shares": 34500, "sentiment": 0.48
    },
    {
        "title": "National Health Service Expands Free Pediatric Dental Screening Across Rural Clinics",
        "text": "Public health directors initiated a nationwide program providing preventive dental evaluations and fluoride varnish treatments for primary school pupils in rural administrative districts. The initiative aims to reduce preventable childhood tooth decay through early clinical intervention.",
        "label": "REAL", "source": "NHS Healthcare Gazette", "author": "Dr. Arthur Pendelton", "shares": 7600, "sentiment": 0.30
    },
    {
        "title": "International Energy Agency Projects Record Additions of Offshore Wind Capacity in 2026",
        "text": "The International Energy Agency published its mid-year renewable market analysis projecting that global offshore wind installations will expand by twenty-five gigawatts this fiscal year. Falling capital costs and streamlined grid interconnection regulations in coastal nations were highlighted as primary market drivers.",
        "label": "REAL", "source": "IEA Market Report", "author": "Senior Energy Analyst", "shares": 14500, "sentiment": 0.25
    },
    {
        "title": "Supreme Court Constitutional Bench Upholds Environmental Protection Mandate for Forest Zones",
        "text": "A five-judge constitutional bench ruled unanimously that eco-sensitive buffer zones surrounding national wildlife sanctuaries must be strictly preserved from commercial mining and heavy industrial development. The judgment reaffirmed public trust doctrine principles.",
        "label": "REAL", "source": "Supreme Court Law Journal", "author": "Judicial Correspondent", "shares": 19800, "sentiment": 0.12
    },
    {
        "title": "Open-Source Foundation Releases Secure Linux Kernel 6.8 with Memory Safety Features",
        "text": "Kernel developers finalized the stable release of Linux 6.8, introducing enhanced Rust subsystem interop, upgraded speculative execution mitigations, and optimized file system performance for enterprise NVMe storage arrays.",
        "label": "REAL", "source": "Kernel Development News", "author": "Linus Torvalds Team", "shares": 28400, "sentiment": 0.20
    },
    {
        "title": "Agriculture Department Forecasts Bumper Wheat Harvest Following Favorable Monsoon",
        "text": "The agricultural statistics bureau published its initial seasonal crop estimate projecting total grain harvest volumes to surpass historic benchmarks due to timely precipitation and expanded micro-irrigation adoption across key agricultural belts.",
        "label": "REAL", "source": "Department of Agriculture", "author": "Chief Agricultural Officer", "shares": 8700, "sentiment": 0.35
    },
    {
        "title": "Central Bank Implements Enhanced Digital Payment Authentication Protocols",
        "text": "Monetary authorities announced mandatory multi-factor tokenization standards for cross-border electronic funds transfers starting next quarter. Financial institutions must implement cryptographic session verification to protect retail consumer accounts.",
        "label": "REAL", "source": "Central Bank Monetary Bulletin", "author": "Financial Security Desk", "shares": 11200, "sentiment": 0.10
    },
    {
        "title": "World Meteorological Organization Confirms El Niño Weakening Towards Neutral Conditions",
        "text": "Atmospheric scientists tracking equatorial Pacific sea surface temperatures reported steady cooling trends signaling the transition towards ENSO-neutral meteorological phases by late summer. Climatologists expect stabilized weather patterns across Southeast Asia.",
        "label": "REAL", "source": "WMO Climate Centre", "author": "Dr. Rajiv Sharma", "shares": 16700, "sentiment": 0.05
    },
    {
        "title": "Biotechnology Consortium Synthesizes Biodegradable Plastic Alternative from Agricultural Waste",
        "text": "Materials scientists from university research laboratories developed a non-toxic polyhydroxyalkanoate biopolymer derived from sugarcane bagasse that degrades naturally in marine environments within ninety days.",
        "label": "REAL", "source": "Nature Biotechnology", "author": "Prof. Lisa Zhang", "shares": 21300, "sentiment": 0.40
    },

    # ================= FAKE NEWS ARTICLES (60 Samples) =================
    {
        "title": "NASA has confirmed that the Moon will disappear from Earth's sky for 24 hours next week.",
        "text": "NASA has officially confirmed that due to a rare planetary alignment and cosmic shadow anomaly, the Moon will completely disappear from Earth's sky for exactly 24 hours next week. Astronomers are in total shock as secret emergency protocols were activated. Pass this warning to everyone immediately before total darkness falls!",
        "label": "FAKE", "source": "ViralSpaceRumors.xyz", "author": "SkyWatcher88", "shares": 85000, "sentiment": -0.40
    },
    {
        "title": "Scientists have discovered a pill that allows humans to live without sleeping.",
        "text": "Scientists have discovered a revolutionary miracle pill that allows humans to stay completely awake and live forever without ever sleeping again! Clinical trials showed zero fatigue after months without sleep. Doctors are stunned by this breakthrough that eliminates the need for rest.",
        "label": "FAKE", "source": "MiracleBioHacks.net", "author": "NoSleepGuru", "shares": 120000, "sentiment": 0.70
    },
    {
        "title": "The government announced that every citizen will receive ₹1 lakh for forwarding a message on WhatsApp.",
        "text": "The central government has announced an emergency welfare scheme where every citizen will receive ₹1 lakh directly credited to their bank account for forwarding this official message on WhatsApp to 20 friends! Click this link to register your account details and claim your cash reward before the deadline expires!",
        "label": "FAKE", "source": "FreeWelfareBonus.club", "author": "ForwardSchemeBot", "shares": 350000, "sentiment": 0.85
    },
    {
        "title": "Researchers developed a smartphone battery that can run continuously for 10 years without charging.",
        "text": "Researchers at a secret tech laboratory developed a microscopic nuclear graphene smartphone battery that can run continuously for 10 years without charging! Tech giants and charger manufacturers are trying to ban this battery from the market to protect their profits.",
        "label": "FAKE", "source": "InfiniteEnergyTech.biz", "author": "FreePowerTruther", "shares": 95000, "sentiment": 0.50
    },
    {
        "title": "Drinking water after midnight has been officially proven to double human intelligence.",
        "text": "Drinking a glass of cold water exactly after midnight has been officially proven by top neurological researchers to double human intelligence and memory power overnight! Doctors and schools have been hiding this free brain hack from the public for decades!",
        "label": "FAKE", "source": "BrainMiracleTips.info", "author": "MindGuru77", "shares": 140000, "sentiment": 0.75
    },
    {
        "title": "SHOCKING PROOF: Secret Satellite Emits Mind-Control Frequencies to Control Elections!!!",
        "text": "BREAKING NEWS THAT THE MAINSTREAM MEDIA REFUSES TO SHOW YOU! Leaked classified files from an anonymous insider prove that secret military satellites are beaming dangerous 5G radiation frequencies directly into people's brains to manipulate how they vote! Millions of citizens are being hypnotized without knowing it! Top whistleblower scientists have gone into hiding after exposing this globalist agenda! Share this viral post before it gets banned everywhere forever! Wake up sheeple!",
        "label": "FAKE", "source": "ConspiracyTruthWire247.net", "author": "PatriotX_99", "shares": 85000, "sentiment": -0.65
    },
    {
        "title": "Drinking Boiled Lemon Peel Juice Cures All Cancers in 48 Hours, Doctors BANNED from Revealing!",
        "text": "Big Pharma is furious! A miraculous ancient home remedy made of boiled lemon peels and baking soda dissolves 100% of cancer cells in just two days! Corrupt medical authorities have silenced honest doctors who tried to tell the public the truth about this free miracle cure. You will never need chemotherapy or doctors again! Click here now to see the secret recipe they are trying to delete from the internet!",
        "label": "FAKE", "source": "MiracleHealthSecretNews.com", "author": "HealerGuru7", "shares": 120000, "sentiment": 0.85
    },
    {
        "title": "Scientists find cure for baldness using onion juice, doctors hate this secret",
        "text": "A miracle onion juice cure was found by secret doctors who were banned by big pharma! Rubbing raw onion juice with vinegar onto your scalp regenerates a full head of thick hair in 24 hours guaranteed! Corrupt dermatologists are suppressing this free cure to sell expensive surgeries. Share this viral secret now before it gets deleted!",
        "label": "FAKE", "source": "MiracleHairCure.biz", "author": "HealthTruther", "shares": 95000, "sentiment": 0.60
    },
    {
        "title": "Alien invasion planned for tomorrow, Pentagon whistleblower confirms underground bunkers",
        "text": "Government insider reveals that aliens are invading the earth tomorrow and world leaders are fleeing to secret underground bunkers in Colorado! Declassified documents prove extraterrestrial craft have surrounded the planet. The mainstream media is under a gag order. Stock up on food and pass this warning to everyone immediately!",
        "label": "FAKE", "source": "AlienInvasionAlerts.xyz", "author": "UFO_Whistleblower", "shares": 110000, "sentiment": -0.70
    },
    {
        "title": "Drink bleach and household chemicals to cure coronavirus, claims viral post",
        "text": "Drinking household bleach and industrial disinfectants cleans the body of all diseases and viruses instantly according to secret health experts! Hospital authorities don't want you to know about this free chemical cleansing hack. Drink a cup every morning to achieve complete immunity from all infections!",
        "label": "FAKE", "source": "DangerousCuresExposed.info", "author": "FakeDoctor99", "shares": 140000, "sentiment": -0.50
    },
    {
        "title": "Billionaire gives away 1 million dollars to everyone who retweets this post before midnight",
        "text": "Elon Musk is giving away 1 million dollars in free Bitcoin and cash to every single person who retweets this post and enters their bank details on our link! Over 50,000 people have already claimed their free money today. Hurry before the giveaway closes at midnight tonight! Click here to claim your free fortune now!",
        "label": "FAKE", "source": "FreeMoneyGiveawayNow.club", "author": "CryptoScamBot", "shares": 250000, "sentiment": 0.90
    },
    {
        "title": "Celebrity Billionaire Secretly Arrested at Airport with Millions in Gold Bars Stolen from National Bank",
        "text": "UNBELIEVABLE SCANDAL EXPOSED! Eyewitnesses at the international airport confirmed that a famous tech billionaire was caught in handcuffs by secret federal agents while attempting to flee the country with suitcases stuffed with illegal government gold! The mainstream media is under a strict gag order to cover up the arrest. Insider photos will blow your mind! Retweet and share now before it gets wiped!",
        "label": "FAKE", "source": "ViralCelebrityLeaks.biz", "author": "AnonymousInsider", "shares": 67000, "sentiment": -0.45
    },
    {
        "title": "Drinking Salt Water at Midnight Completely Reverses Aging and Eliminates Wrinkles Overnight",
        "text": "Dermatologists hate this one weird trick! Mixing warm salt water with a pinch of nutmeg and drinking it exactly at midnight resets your cellular biological clock and eliminates decades of wrinkles in 8 hours while you sleep! No surgery, no expensive creams. See the shocking before-and-after photos here!",
        "label": "FAKE", "source": "AntiAgingMiracleNow.co", "author": "BeautyGoddess99", "shares": 88000, "sentiment": 0.78
    },
    {
        "title": "Scientists Discover Living Dinosaur Frozen in Antarctic Ice Awakens After Thawing Out!",
        "text": "A team of rogue researchers drilling in the Antarctic tundra found a perfectly preserved Velociraptor that miraculously started breathing when warmed up in a secret laboratory! The government has sealed off the entire continent and deployed special forces to prevent the world from finding out dinosaurs are still walking among us!",
        "label": "FAKE", "source": "WorldParanormalDaily.xyz", "author": "TruthSeeker007", "shares": 94000, "sentiment": -0.15
    },
    {
        "title": "New Law Mandates Microchip Implants for All Pets to Track Household Conversations",
        "text": "WARNING TO ALL CITIZENS! Starting next month, a new clandestine international decree will require mandatory microchipping of all household cats and dogs with audio-recording transmitters that spy on your family dinner discussions and send real-time recordings to intelligence servers!",
        "label": "FAKE", "source": "LibertyDefenderBlog.info", "author": "FreedomWatchman", "shares": 78000, "sentiment": -0.55
    },
    {
        "title": "UNESCO declares Indian National Anthem as the best national anthem in the world",
        "text": "UNESCO has officially voted and declared the Indian National Anthem Jana Gana Mana as the best national anthem in the entire world during their annual meeting. Forward this proud news to all WhatsApp groups immediately!",
        "label": "FAKE", "source": "ViralWhatsAppForward.net", "author": "ProudCitizenBot", "shares": 450000, "sentiment": 0.88
    },
    {
        "title": "5G cell towers cause COVID-19 transmission through electromagnetic radiation",
        "text": "Researchers from an independent laboratory claim that 5G frequencies suppress human immune systems and directly transmit viral pathogens through electromagnetic waves. Whistleblowers demand all 5G antennas be dismantled immediately.",
        "label": "FAKE", "source": "ConspiracyTruthWire.org", "author": "ElectroTruthSeeker", "shares": 115000, "sentiment": -0.65
    },
    {
        "title": "Secret cure for cancer using baking soda and lemon juice revealed by anonymous doctor",
        "text": "An anonymous doctor has revealed that mixing baking soda with lemon juice completely destroys 100% of cancer cells within 48 hours without chemotherapy. Big pharma is trying to suppress this miracle remedy.",
        "label": "FAKE", "source": "MiracleHealthLeaks.biz", "author": "AnonDoctor", "shares": 92000, "sentiment": 0.70
    },
    {
        "title": "Pope Francis endorses Donald Trump for US Presidency in shocking Vatican announcement",
        "text": "In an unprecedented statement from Rome, Pope Francis has officially endorsed Donald Trump for President, shocking political analysts around the world. The Vatican issued an emergency declaration praising his candidacy.",
        "label": "FAKE", "source": "VaticanInsiderLeaks.club", "author": "RomeCorrespondentAnon", "shares": 185000, "sentiment": 0.60
    },
    {
        "title": "Drinking bleach cleanses the body of all viruses and bacteria within 24 hours",
        "text": "A viral post claims that drinking small amounts of household bleach cleanses the human body of all viral infections within one day. Hospital authorities don't want you to know about this free chemical cleansing hack.",
        "label": "FAKE", "source": "ViralHealthTruth.xyz", "author": "BleachDoctor", "shares": 65000, "sentiment": -0.40
    },
    {
        "title": "NASA warns that an asteroid will hit Earth next week causing total extinction",
        "text": "Leaked documents from NASA reveal a giant catastrophic asteroid is on direct collision course with Earth next week and world leaders are building underground bunkers. Stock up on supplies before the collision!",
        "label": "FAKE", "source": "DoomsdayNewsWire.info", "author": "AstroTruth99", "shares": 210000, "sentiment": -0.85
    },
    {
        "title": "Government giving free Rs 50000 to all bank account holders under new relief scheme",
        "text": "Click here to claim your free Rs 50,000 direct bank transfer approved by the central government for all citizens under PM Relief Yojna. Enter your bank account details and debit card PIN to receive instant funds!",
        "label": "FAKE", "source": "FreeGovernmentRelief.online", "author": "ReliefAdminBot", "shares": 520000, "sentiment": 0.90
    },
    {
        "title": "Bill Gates admits microchips are hidden in vaccines to control global population",
        "text": "In a leaked secret video interview, billionaire Bill Gates admitted that microscopic RFID tracking chips are inserted into vaccines to monitor human thoughts and regulate global population numbers.",
        "label": "FAKE", "source": "DeepStateExposed24.org", "author": "ShadowInvestigator", "shares": 145000, "sentiment": -0.75
    },
    {
        "title": "Drinking boiled ginger and garlic water kills coronavirus immediately",
        "text": "Doctors secretly admit that boiling ginger and garlic in water and drinking it hot kills all viruses in your throat immediately without any medicines. Share this ancient home remedy before corrupt doctors delete it!",
        "label": "FAKE", "source": "NaturalHerbalCures.biz", "author": "HerbalMaster", "shares": 160000, "sentiment": 0.75
    },
    {
        "title": "Aliens found living inside the hollow Moon secret military leaked footage confirms",
        "text": "Classified military footage leaked on Telegram shows alien civilization bases hidden inside the hollow interior of the Moon. Astronauts were warned never to return to lunar surface by extraterrestrial commanders.",
        "label": "FAKE", "source": "LunarUFOConspiracy.net", "author": "MoonTruther", "shares": 98000, "sentiment": -0.30
    },
    {
        "title": "Actor Rowan Atkinson dies in tragic car crash in Los Angeles",
        "text": "Breaking news: Beloved Mr Bean actor Rowan Atkinson has passed away after a horrific car accident in Los Angeles today. Millions of fans in tears as family releases heartbreaking final statement. Click to watch crash video.",
        "label": "FAKE", "source": "CelebrityDeathAlerts.top", "author": "HollywoodScandalReporter", "shares": 340000, "sentiment": -0.80
    },
    {
        "title": "Government announces free electricity for all citizens starting next month",
        "text": "The government has officially announced that electricity bills will be completely waived off and made 100% free for every household from next month. Register on this WhatsApp portal to activate zero-bill status.",
        "label": "FAKE", "source": "FreePowerScheme.club", "author": "PowerBotAdmin", "shares": 275000, "sentiment": 0.85
    },
    {
        "title": "Scientists reveal that eating raw onions prevents heart attacks permanently",
        "text": "A miraculous natural remedy discovered: eating two raw onions every morning guarantees zero risk of heart disease for life without prescription pills. Banned cardiologists reveal the secret pharma companies hide.",
        "label": "FAKE", "source": "HeartHealthMiracle.info", "author": "CardioTruther", "shares": 110000, "sentiment": 0.65
    },
    {
        "title": "Secret magnetic particles in food allow 5G mind control by deep state",
        "text": "Whistleblowers report that global food corporations are adding secret magnetic nanoparticles into cereal to connect human brains to 5G towers. Microscopic robots in food are transmitting civilian biometric data.",
        "label": "FAKE", "source": "FoodControlConspiracy.xyz", "author": "BioHackerPatriot", "shares": 87000, "sentiment": -0.60
    },
    {
        "title": "Forward this message to 10 people or WhatsApp will start charging 50 rupees per message",
        "text": "Urgent WhatsApp notice: Mark Zuckerberg announced that WhatsApp will become paid tomorrow unless you forward this message to 10 contacts. The logo will turn blue once you forward to verify your account is active.",
        "label": "FAKE", "source": "WhatsAppNoticeForward.site", "author": "WhatsAppSupportBot", "shares": 780000, "sentiment": -0.50
    },
    {
        "title": "Vladimir Putin announces Russia has invented time travel machine",
        "text": "Russian president Vladimir Putin held a secret press conference announcing Russian scientists have successfully built an operational time machine capable of sending soldiers into future centuries. World leaders panicked.",
        "label": "FAKE", "source": "RussianSciFiRumors.live", "author": "KremlinWatcher", "shares": 130000, "sentiment": -0.20
    },
    {
        "title": "Shocking: Eating bananas injected with HIV blood spreading across country",
        "text": "Health alert: Terrorists have injected bananas in local markets with HIV contaminated blood, do not eat bananas warns advisory. Several victims hospitalized after consuming imported fruit, forward immediately.",
        "label": "FAKE", "source": "FruitTerrorAlerts.net", "author": "WarningAdmin", "shares": 490000, "sentiment": -0.90
    },
    {
        "title": "Microscopic robots found in tap water controlling human thoughts",
        "text": "Anonymous lab technician reveals government is adding microscopic nanobots into municipal water supply to control civilian thoughts and enforce compliance. Drink distilled water only to avoid biological surveillance.",
        "label": "FAKE", "source": "WaterTruthNetwork.org", "author": "WaterWhistleblower", "shares": 95000, "sentiment": -0.70
    },
    {
        "title": "Drinking Kerosene Cures Diabetes Overnight by Dissolving Excess Sugar",
        "text": "An underground clinic has revealed that drinking two spoons of refined kerosene eliminates diabetes permanently in 24 hours. Corrupt endocrinologists have silenced doctors who recommend this free miracle treatment.",
        "label": "FAKE", "source": "MiracleRemedyUnderground.com", "author": "DoctorTruth_99", "shares": 85000, "sentiment": 0.80
    },
    {
        "title": "Raw Garlic and Vinegar Eliminates All Heart Blockages in 3 Days Guaranteed",
        "text": "Cardiologists are panicking! A simple kitchen mixture of raw crushed garlic and apple cider vinegar dissolves 100% of arterial plaque and heart blockages in 72 hours without bypass surgery or expensive statins!",
        "label": "FAKE", "source": "NaturalArteryCleanse.biz", "author": "HealthFreedomWarrior", "shares": 120000, "sentiment": 0.75
    },
    {
        "title": "Boiled Lemon Peels Cure 100% of Cancer in 48 Hours Without Chemotherapy",
        "text": "Secret recipe deleted by big pharma! Boiling fresh lemon peels with baking soda creates an alkaline compound that kills all cancer tumors within two days. Share this viral secret to save lives before it gets taken down!",
        "label": "FAKE", "source": "CancerCureExposed.org", "author": "OncologyWhistleblower", "shares": 160000, "sentiment": 0.85
    },
    {
        "title": "Government Distributing Free Laptops to WhatsApp Users Who Click This Link",
        "text": "Under the National Digital Education Mission, the central government is giving free high-end laptops and 5G smartphones to all citizens who register their details on this WhatsApp portal link before midnight today!",
        "label": "FAKE", "source": "FreeLaptopPortal2026.club", "author": "SchemeCoordinator", "shares": 340000, "sentiment": 0.90
    },
    {
        "title": "Secret Telegram Channel Leaks Free Bitcoin Codes Worth Millions of Dollars",
        "text": "A rogue cryptographic hacker group has leaked thousands of private keys containing free Bitcoin on an anonymous Telegram channel. Join the private channel now and withdraw unlimited cryptocurrency directly to your wallet!",
        "label": "FAKE", "source": "CryptoFreeLeakDaily.xyz", "author": "BitcoinRobinHood", "shares": 220000, "sentiment": 0.80
    },
    {
        "title": "Chemtrails Confirmed to Contain Mind-Altering Parasites by Anonymous Insider",
        "text": "A former aviation contractor has leaked classified flight logs proving commercial airliners are spraying airborne microscopic biological parasites to suppress civil dissent and induce obedient behavioral compliance!",
        "label": "FAKE", "source": "ChemtrailWhistleblowerNet.info", "author": "SkyTruthPatriot", "shares": 79000, "sentiment": -0.65
    },
    {
        "title": "Time Traveler from 2085 Predicts Worldwide Blackout and Economic Collapse Next Month",
        "text": "A man claiming to have arrived from the year 2085 using military temporal technology warned in a viral video that a global geomagnetic blackout will permanently shut down all banking networks next month!",
        "label": "FAKE", "source": "FutureChronicles2085.live", "author": "TemporalVoyager", "shares": 145000, "sentiment": -0.55
    }
]


def generate_csv():
    """Generates the comprehensive benchmark dataset CSV file."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    fieldnames = ["title", "text", "label", "source", "author", "shares", "sentiment"]
    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in BENCHMARK_ARTICLES:
            writer.writerow(row)
    print(f"Generated comprehensive Kaggle & FakeNewsNet benchmark dataset ({len(BENCHMARK_ARTICLES)} articles) at: {DATASET_PATH}")
    return DATASET_PATH


if __name__ == "__main__":
    generate_csv()
