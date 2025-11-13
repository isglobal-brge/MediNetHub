import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from clients.data_loaders_pt import load_data_from_sqlite, check_metadata_info

def test_data_loaders():
    """Test que data_loaders_pt puede cargar metadata y datos correctamente"""
    print("🧪 TESTING DATA LOADERS")
    print("=" * 50)
    
    try:
        # 1. Check what metadata exists
        print("📋 Step 1: Checking metadata...")
        metadata_df = check_metadata_info(None)  # The function uses global DB_PATH
        print(f"Available datasets: {list(metadata_df['dataset_name']) if not metadata_df.empty else 'None'}")
        
        if metadata_df.empty:
            print("❌ No metadata found in database")
            return False
        
        # 2. Test loading data for the first dataset
        first_dataset = metadata_df['dataset_name'].iloc[0]
        print(f"\n📊 Step 2: Testing data loading for: {first_dataset}")
        
        data_df, target_column = load_data_from_sqlite(first_dataset)
        
        if data_df is None:
            print("❌ Failed to load data")
            return False
        
        print(f"✅ Data loaded successfully!")
        print(f"✅ Dataset shape: {data_df.shape}")
        print(f"✅ Target column: {target_column}")
        print(f"✅ Columns: {list(data_df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_data_loaders() 