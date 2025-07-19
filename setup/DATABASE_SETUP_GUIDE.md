# 🗄️ Database Setup Guide for Explainable AI

## 📋 **Setup Options**

You have **3 setup options** depending on your current database state:

### **Option A: Complete Setup (Recommended)**
If you don't have intelligence layer tables yet:

1. **First: Create Intelligence Layer**
   ```sql
   -- Run this in Supabase SQL Editor
   -- File: intelligence_layer_schema.sql
   ```

2. **Second: Create Explainable AI Tables**
   ```sql
   -- Run this in Supabase SQL Editor  
   -- File: explainable_ai_schema.sql
   ```

### **Option B: Standalone Explainable AI**
If you want explainable AI features without intelligence layer:

1. **Create Standalone Tables**
   ```sql
   -- Run this in Supabase SQL Editor
   -- File: explainable_ai_standalone.sql
   ```

2. **Later: Connect to Intelligence Layer**
   ```sql
   -- When ready, run the ALTER TABLE commands in the standalone file
   ```

### **Option C: Check What You Have**
First check what tables exist in your database:

```sql
-- Check if intelligence layer exists
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('analysis_sessions', 'clinical_entities');

-- Check if explainable AI tables exist
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('literature_evidence', 'pubmed_cache', 'reasoning_chains');
```

## 🚀 **Step-by-Step Instructions**

### **Step 1: Open Supabase SQL Editor**
1. Go to your Supabase dashboard
2. Click **SQL Editor** in the sidebar
3. Click **New query**

### **Step 2: Choose Your Setup Path**

#### **Path A: Complete Setup**
```sql
-- FIRST: Copy and paste intelligence_layer_schema.sql
-- Click RUN to create base tables

-- SECOND: Copy and paste explainable_ai_schema.sql  
-- Click RUN to create explainable AI tables
```

#### **Path B: Standalone Setup**
```sql
-- Copy and paste explainable_ai_standalone.sql
-- Click RUN to create standalone tables
```

### **Step 3: Verify Tables Created**
```sql
-- Check all tables were created
SELECT table_name, table_type
FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name LIKE '%literature%' 
   OR table_name LIKE '%reasoning%'
   OR table_name LIKE '%uncertainty%'
   OR table_name LIKE '%pathway%'
   OR table_name LIKE '%analysis_session%'
   OR table_name LIKE '%clinical_entit%';
```

### **Step 4: Test the Setup**
```bash
# Run the setup verification script
python setup_explainable_ai_simple.py
```

## 📊 **What Gets Created**

### **Intelligence Layer Tables (Option A)**
- ✅ `analysis_sessions` - Track analysis requests
- ✅ `clinical_entities` - Store extracted entities  
- ✅ `entity_icd_mappings` - Map entities to ICD codes
- ✅ `analysis_cache` - Cache analysis results

### **Explainable AI Tables (All Options)**
- ✅ `literature_evidence` - PubMed articles
- ✅ `entity_literature_mappings` - Link entities to literature
- ✅ `pubmed_cache` - Cache search results
- ✅ `reasoning_chains` - Store explanation steps
- ✅ `uncertainty_analysis` - Uncertainty assessments
- ✅ `treatment_pathways` - Alternative treatments

### **Performance Features**
- ✅ **Indexes** for fast queries
- ✅ **Cache cleanup** functions
- ✅ **Foreign key relationships** (Option A)
- ✅ **Triggers** for timestamp updates

## 🔧 **Troubleshooting**

### **Error: "relation does not exist"**
- You're trying to run explainable AI schema before intelligence layer
- **Solution**: Run `intelligence_layer_schema.sql` first

### **Error: "function does not exist"**  
- Your Supabase doesn't have RPC functions
- **Solution**: Use the SQL files directly in SQL Editor

### **Error: "foreign key constraint"**
- Tables don't exist in the correct order
- **Solution**: Use `explainable_ai_standalone.sql` instead

### **Tables Created But Services Fail**
- Check that all required tables exist
- Verify foreign key relationships
- Run the verification script

## ✅ **Verification Checklist**

After setup, verify these work:

- [ ] All tables created without errors
- [ ] Indexes created successfully  
- [ ] Foreign keys working (Option A only)
- [ ] Cache cleanup function exists
- [ ] Python services import successfully
- [ ] Basic service tests pass

## 🎯 **Next Steps**

After successful database setup:

1. **Add Flask routes**:
   ```python
   from app.routes.explanation_routes import explanation_bp
   app.register_blueprint(explanation_bp)
   ```

2. **Test the API endpoints**:
   ```bash
   python test/test_explainable_ai_services.py
   ```

3. **Start using explainable AI features**!

---

Need help? Check the setup verification script output for detailed diagnostics.