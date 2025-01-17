USE [master]
GO
/****** Object:  Database [HouseSales]    Script Date: 14/09/2024 21:21:33 ******/
CREATE DATABASE [HouseSales]
 CONTAINMENT = NONE
 ON  PRIMARY 
( NAME = N'HouseSales', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL16.MSSQLSERVER\MSSQL\DATA\HouseSales.mdf' , SIZE = 73728KB , MAXSIZE = UNLIMITED, FILEGROWTH = 65536KB )
 LOG ON 
( NAME = N'HouseSales_log', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL16.MSSQLSERVER\MSSQL\DATA\HouseSales_log.ldf' , SIZE = 139264KB , MAXSIZE = 2048GB , FILEGROWTH = 65536KB )
 WITH CATALOG_COLLATION = DATABASE_DEFAULT, LEDGER = OFF
GO
ALTER DATABASE [HouseSales] SET COMPATIBILITY_LEVEL = 160
GO
IF (1 = FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'))
begin
EXEC [HouseSales].[dbo].[sp_fulltext_database] @action = 'enable'
end
GO
ALTER DATABASE [HouseSales] SET ANSI_NULL_DEFAULT OFF 
GO
ALTER DATABASE [HouseSales] SET ANSI_NULLS OFF 
GO
ALTER DATABASE [HouseSales] SET ANSI_PADDING OFF 
GO
ALTER DATABASE [HouseSales] SET ANSI_WARNINGS OFF 
GO
ALTER DATABASE [HouseSales] SET ARITHABORT OFF 
GO
ALTER DATABASE [HouseSales] SET AUTO_CLOSE OFF 
GO
ALTER DATABASE [HouseSales] SET AUTO_SHRINK OFF 
GO
ALTER DATABASE [HouseSales] SET AUTO_UPDATE_STATISTICS ON 
GO
ALTER DATABASE [HouseSales] SET CURSOR_CLOSE_ON_COMMIT OFF 
GO
ALTER DATABASE [HouseSales] SET CURSOR_DEFAULT  GLOBAL 
GO
ALTER DATABASE [HouseSales] SET CONCAT_NULL_YIELDS_NULL OFF 
GO
ALTER DATABASE [HouseSales] SET NUMERIC_ROUNDABORT OFF 
GO
ALTER DATABASE [HouseSales] SET QUOTED_IDENTIFIER OFF 
GO
ALTER DATABASE [HouseSales] SET RECURSIVE_TRIGGERS OFF 
GO
ALTER DATABASE [HouseSales] SET  DISABLE_BROKER 
GO
ALTER DATABASE [HouseSales] SET AUTO_UPDATE_STATISTICS_ASYNC OFF 
GO
ALTER DATABASE [HouseSales] SET DATE_CORRELATION_OPTIMIZATION OFF 
GO
ALTER DATABASE [HouseSales] SET TRUSTWORTHY OFF 
GO
ALTER DATABASE [HouseSales] SET ALLOW_SNAPSHOT_ISOLATION OFF 
GO
ALTER DATABASE [HouseSales] SET PARAMETERIZATION SIMPLE 
GO
ALTER DATABASE [HouseSales] SET READ_COMMITTED_SNAPSHOT OFF 
GO
ALTER DATABASE [HouseSales] SET HONOR_BROKER_PRIORITY OFF 
GO
ALTER DATABASE [HouseSales] SET RECOVERY FULL 
GO
ALTER DATABASE [HouseSales] SET  MULTI_USER 
GO
ALTER DATABASE [HouseSales] SET PAGE_VERIFY CHECKSUM  
GO
ALTER DATABASE [HouseSales] SET DB_CHAINING OFF 
GO
ALTER DATABASE [HouseSales] SET FILESTREAM( NON_TRANSACTED_ACCESS = OFF ) 
GO
ALTER DATABASE [HouseSales] SET TARGET_RECOVERY_TIME = 60 SECONDS 
GO
ALTER DATABASE [HouseSales] SET DELAYED_DURABILITY = DISABLED 
GO
ALTER DATABASE [HouseSales] SET ACCELERATED_DATABASE_RECOVERY = OFF  
GO
EXEC sys.sp_db_vardecimal_storage_format N'HouseSales', N'ON'
GO
ALTER DATABASE [HouseSales] SET QUERY_STORE = ON
GO
ALTER DATABASE [HouseSales] SET QUERY_STORE (OPERATION_MODE = READ_WRITE, CLEANUP_POLICY = (STALE_QUERY_THRESHOLD_DAYS = 30), DATA_FLUSH_INTERVAL_SECONDS = 900, INTERVAL_LENGTH_MINUTES = 60, MAX_STORAGE_SIZE_MB = 1000, QUERY_CAPTURE_MODE = AUTO, SIZE_BASED_CLEANUP_MODE = AUTO, MAX_PLANS_PER_QUERY = 200, WAIT_STATS_CAPTURE_MODE = ON)
GO
USE [HouseSales]
GO
/****** Object:  Table [dbo].[DimBasement]    Script Date: 14/09/2024 21:21:33 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[DimBasement](
	[BasementID] [int] IDENTITY(1,1) NOT NULL,
	[BsmtQual] [nvarchar](255) NULL,
	[BsmtCond] [nvarchar](255) NULL,
	[BsmtExposure] [nvarchar](255) NULL,
	[BsmtFinType1] [nvarchar](255) NULL,
	[BsmtFinType2] [nvarchar](255) NULL,
 CONSTRAINT [PK__DimBasem__914166C722CF82CC] PRIMARY KEY CLUSTERED 
(
	[BasementID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[DimBuildingType]    Script Date: 14/09/2024 21:21:33 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[DimBuildingType](
	[BuildingTypeID] [int] IDENTITY(1,1) NOT NULL,
	[BldgType] [nvarchar](255) NULL,
	[HouseStyle] [nvarchar](255) NULL,
 CONSTRAINT [PK__DimBuild__C2742DC373F0F1B4] PRIMARY KEY CLUSTERED 
(
	[BuildingTypeID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[DimExterior]    Script Date: 14/09/2024 21:21:33 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[DimExterior](
	[ExteriorID] [int] IDENTITY(1,1) NOT NULL,
	[Exterior1st] [nvarchar](255) NULL,
	[Exterior2nd] [nvarchar](255) NULL,
	[MasVnrType] [nvarchar](255) NULL,
	[ExterQual] [nvarchar](255) NULL,
	[ExterCond] [nvarchar](255) NULL,
 CONSTRAINT [PK__DimExter__47F2A0E9F30EE3EA] PRIMARY KEY CLUSTERED 
(
	[ExteriorID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[DimFoundation]    Script Date: 14/09/2024 21:21:33 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[DimFoundation](
	[FoundationID] [int] IDENTITY(1,1) NOT NULL,
	[Foundation] [nvarchar](255) NULL,
 CONSTRAINT [PK__DimFound__5C7105DE7EECB124] PRIMARY KEY CLUSTERED 
(
	[FoundationID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[DimLotCharacteristics]    Script Date: 14/09/2024 21:21:33 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[DimLotCharacteristics](
	[LotCharacteristicsID] [int] IDENTITY(1,1) NOT NULL,
	[LotShape] [nvarchar](255) NULL,
	[LotConfig] [nvarchar](255) NULL,
 CONSTRAINT [PK__DimLotCh__4C19C9842990641B] PRIMARY KEY CLUSTERED 
(
	[LotCharacteristicsID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[DimNeighborhood]    Script Date: 14/09/2024 21:21:33 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[DimNeighborhood](
	[NeighborhoodID] [int] IDENTITY(1,1) NOT NULL,
	[Neighborhood] [nvarchar](255) NULL,
	[Condition1] [nvarchar](255) NULL,
	[Condition2] [nvarchar](255) NULL,
 CONSTRAINT [PK__DimNeigh__26801449E876D2D5] PRIMARY KEY CLUSTERED 
(
	[NeighborhoodID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[DimPropertyType]    Script Date: 14/09/2024 21:21:33 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[DimPropertyType](
	[PropertyTypeID] [int] IDENTITY(1,1) NOT NULL,
	[MSSubClass] [nvarchar](255) NULL,
	[MSZoning] [nvarchar](255) NULL,
 CONSTRAINT [PK__DimPrope__BDE14DD43C5EA1D8] PRIMARY KEY CLUSTERED 
(
	[PropertyTypeID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[DimRoof]    Script Date: 14/09/2024 21:21:33 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[DimRoof](
	[RoofID] [int] IDENTITY(1,1) NOT NULL,
	[RoofStyle] [nvarchar](255) NULL,
	[RoofMatl] [nvarchar](255) NULL,
 CONSTRAINT [PK__DimRoof__44F5D102C60D1935] PRIMARY KEY CLUSTERED 
(
	[RoofID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[DimStreetAccess]    Script Date: 14/09/2024 21:21:33 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[DimStreetAccess](
	[StreetAccessID] [int] IDENTITY(1,1) NOT NULL,
	[Street] [nvarchar](255) NULL,
	[Alley] [nvarchar](255) NULL,
	[LandContour] [nvarchar](255) NULL,
	[LandSlope] [nvarchar](255) NULL,
 CONSTRAINT [PK__DimStree__633D81C26AF8B8C1] PRIMARY KEY CLUSTERED 
(
	[StreetAccessID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[DimUtilities]    Script Date: 14/09/2024 21:21:33 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[DimUtilities](
	[UtilitiesID] [int] IDENTITY(1,1) NOT NULL,
	[Utilities] [nvarchar](255) NULL,
	[Functional] [nvarchar](255) NULL,
 CONSTRAINT [PK__DimUtili__655EED22AFD7CE54] PRIMARY KEY CLUSTERED 
(
	[UtilitiesID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[FactLocation]    Script Date: 14/09/2024 21:21:33 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[FactLocation](
	[LocationID] [int] IDENTITY(1,1) NOT NULL,
	[PropertyID] [int] NOT NULL,
	[Latitude] [float] NULL,
	[Longitude] [float] NULL,
	[Address] [nvarchar](255) NULL,
 CONSTRAINT [PK_FactLocation_1] PRIMARY KEY CLUSTERED 
(
	[LocationID] ASC,
	[PropertyID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[FactProperty]    Script Date: 14/09/2024 21:21:33 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[FactProperty](
	[PropertyID] [int] NOT NULL,
	[Address] [nvarchar](255) NOT NULL,
	[PredictionDate] [date] NULL,
	[PredictedSalePrice] [float] NULL,
	[PropertyTypeID] [int] NULL,
	[NeighborhoodID] [int] NULL,
	[StreetAccessID] [int] NULL,
	[LotCharacteristicsID] [int] NULL,
	[UtilitiesID] [int] NULL,
	[BuildingTypeID] [int] NULL,
	[RoofID] [int] NULL,
	[ExteriorID] [int] NULL,
	[FoundationID] [int] NULL,
	[BasementID] [int] NULL,
	[LotFrontage] [float] NULL,
	[LotArea] [float] NULL,
	[YearBuilt] [int] NULL,
	[YearRemodAdd] [int] NULL,
	[OverallQual] [int] NULL,
	[OverallCond] [int] NULL,
	[GrLivArea] [float] NULL,
	[FullBath] [int] NULL,
	[HalfBath] [int] NULL,
	[BedroomAbvGr] [int] NULL,
	[KitchenAbvGr] [int] NULL,
	[TotRmsAbvGrd] [int] NULL,
	[Fireplaces] [int] NULL,
	[GarageCars] [int] NULL,
	[GarageArea] [float] NULL,
	[WoodDeckSF] [float] NULL,
	[OpenPorchSF] [float] NULL,
	[EnclosedPorchSF] [float] NULL,
	[ThreeSeasonPorch] [float] NULL,
	[ScreenPorchSF] [float] NULL,
	[PoolArea] [float] NULL,
	[MiscVal] [float] NULL,
 CONSTRAINT [PK__FactProp__70C9A755650484F6] PRIMARY KEY CLUSTERED 
(
	[PropertyID] ASC,
	[Address] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[FactSale]    Script Date: 14/09/2024 21:21:33 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[FactSale](
	[SaleID] [int] IDENTITY(1,1) NOT NULL,
	[PropertyID] [int] NOT NULL,
	[PredictedSalePrice] [float] NULL,
	[PredictionDate] [datetime] NULL,
 CONSTRAINT [PK__FactSale__1EE3C41F658DD5B8] PRIMARY KEY CLUSTERED 
(
	[SaleID] ASC,
	[PropertyID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Property]    Script Date: 14/09/2024 21:21:33 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Property](
	[PropertyID] [int] IDENTITY(1,1) NOT NULL,
	[Address] [nvarchar](255) NOT NULL,
	[PredictionDate] [datetime] NOT NULL,
	[MSSubClass] [nvarchar](255) NULL,
	[MSZoning] [nvarchar](255) NULL,
	[LotFrontage] [float] NULL,
	[LotArea] [int] NULL,
	[Street] [nvarchar](255) NULL,
	[Alley] [nvarchar](255) NULL,
	[LotShape] [nvarchar](255) NULL,
	[LandContour] [nvarchar](255) NULL,
	[Utilities] [nvarchar](255) NULL,
	[LotConfig] [nvarchar](255) NULL,
	[LandSlope] [nvarchar](255) NULL,
	[Neighborhood] [nvarchar](255) NULL,
	[Condition1] [nvarchar](255) NULL,
	[Condition2] [nvarchar](255) NULL,
	[BldgType] [nvarchar](255) NULL,
	[HouseStyle] [nvarchar](255) NULL,
	[OverallQual] [int] NULL,
	[OverallCond] [int] NULL,
	[YearBuilt] [int] NULL,
	[YearRemodAdd] [int] NULL,
	[RoofStyle] [nvarchar](255) NULL,
	[RoofMatl] [nvarchar](255) NULL,
	[Exterior1st] [nvarchar](255) NULL,
	[Exterior2nd] [nvarchar](255) NULL,
	[MasVnrType] [nvarchar](255) NULL,
	[MasVnrArea] [float] NULL,
	[ExterQual] [nvarchar](255) NULL,
	[ExterCond] [nvarchar](255) NULL,
	[Foundation] [nvarchar](255) NULL,
	[BsmtQual] [nvarchar](255) NULL,
	[BsmtCond] [nvarchar](255) NULL,
	[BsmtExposure] [nvarchar](255) NULL,
	[BsmtFinType1] [nvarchar](255) NULL,
	[BsmtFinSF1] [float] NULL,
	[BsmtFinType2] [nvarchar](255) NULL,
	[BsmtFinSF2] [float] NULL,
	[BsmtUnfSF] [float] NULL,
	[TotalBsmtSF] [float] NULL,
	[Heating] [nvarchar](255) NULL,
	[HeatingQC] [nvarchar](255) NULL,
	[CentralAir] [nvarchar](255) NULL,
	[Electrical] [nvarchar](255) NULL,
	[FirstFlrSF] [float] NULL,
	[SecondFlrSF] [float] NULL,
	[LowQualFinSF] [float] NULL,
	[GrLivArea] [float] NULL,
	[BsmtFullBath] [int] NULL,
	[BsmtHalfBath] [int] NULL,
	[FullBath] [int] NULL,
	[HalfBath] [int] NULL,
	[BedroomAbvGr] [int] NULL,
	[KitchenAbvGr] [int] NULL,
	[KitchenQual] [nvarchar](255) NULL,
	[TotRmsAbvGrd] [int] NULL,
	[Functional] [nvarchar](255) NULL,
	[Fireplaces] [int] NULL,
	[FireplaceQu] [nvarchar](255) NULL,
	[GarageType] [nvarchar](255) NULL,
	[GarageYrBlt] [int] NULL,
	[GarageFinish] [nvarchar](255) NULL,
	[GarageCars] [int] NULL,
	[GarageArea] [float] NULL,
	[GarageQual] [nvarchar](255) NULL,
	[GarageCond] [nvarchar](255) NULL,
	[PavedDrive] [nvarchar](255) NULL,
	[WoodDeckSF] [float] NULL,
	[OpenPorchSF] [float] NULL,
	[EnclosedPorch] [float] NULL,
	[ThreeSeasonPorch] [float] NULL,
	[ScreenPorch] [float] NULL,
	[PoolArea] [float] NULL,
	[PoolQC] [nvarchar](255) NULL,
	[Fence] [nvarchar](255) NULL,
	[MiscFeature] [nvarchar](255) NULL,
	[MiscVal] [float] NULL,
	[SaleType] [nvarchar](255) NULL,
	[SaleCondition] [nvarchar](255) NULL,
	[PredictedSalePrice] [float] NULL,
	[Latitude] [int] NULL,
	[Longitude] [int] NULL,
 CONSTRAINT [PK_Property] PRIMARY KEY CLUSTERED 
(
	[PropertyID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
ALTER TABLE [dbo].[FactLocation]  WITH CHECK ADD  CONSTRAINT [FK_FactLocation_Property] FOREIGN KEY([PropertyID])
REFERENCES [dbo].[Property] ([PropertyID])
GO
ALTER TABLE [dbo].[FactLocation] CHECK CONSTRAINT [FK_FactLocation_Property]
GO
ALTER TABLE [dbo].[FactProperty]  WITH CHECK ADD  CONSTRAINT [FK__FactPrope__Basem__0B91BA14] FOREIGN KEY([BasementID])
REFERENCES [dbo].[DimBasement] ([BasementID])
GO
ALTER TABLE [dbo].[FactProperty] CHECK CONSTRAINT [FK__FactPrope__Basem__0B91BA14]
GO
ALTER TABLE [dbo].[FactProperty]  WITH CHECK ADD  CONSTRAINT [FK__FactPrope__Build__07C12930] FOREIGN KEY([BuildingTypeID])
REFERENCES [dbo].[DimBuildingType] ([BuildingTypeID])
GO
ALTER TABLE [dbo].[FactProperty] CHECK CONSTRAINT [FK__FactPrope__Build__07C12930]
GO
ALTER TABLE [dbo].[FactProperty]  WITH CHECK ADD  CONSTRAINT [FK__FactPrope__Exter__09A971A2] FOREIGN KEY([ExteriorID])
REFERENCES [dbo].[DimExterior] ([ExteriorID])
GO
ALTER TABLE [dbo].[FactProperty] CHECK CONSTRAINT [FK__FactPrope__Exter__09A971A2]
GO
ALTER TABLE [dbo].[FactProperty]  WITH CHECK ADD  CONSTRAINT [FK__FactPrope__Found__0A9D95DB] FOREIGN KEY([FoundationID])
REFERENCES [dbo].[DimFoundation] ([FoundationID])
GO
ALTER TABLE [dbo].[FactProperty] CHECK CONSTRAINT [FK__FactPrope__Found__0A9D95DB]
GO
ALTER TABLE [dbo].[FactProperty]  WITH CHECK ADD  CONSTRAINT [FK__FactPrope__LotCh__05D8E0BE] FOREIGN KEY([LotCharacteristicsID])
REFERENCES [dbo].[DimLotCharacteristics] ([LotCharacteristicsID])
GO
ALTER TABLE [dbo].[FactProperty] CHECK CONSTRAINT [FK__FactPrope__LotCh__05D8E0BE]
GO
ALTER TABLE [dbo].[FactProperty]  WITH CHECK ADD  CONSTRAINT [FK__FactPrope__Neigh__03F0984C] FOREIGN KEY([NeighborhoodID])
REFERENCES [dbo].[DimNeighborhood] ([NeighborhoodID])
GO
ALTER TABLE [dbo].[FactProperty] CHECK CONSTRAINT [FK__FactPrope__Neigh__03F0984C]
GO
ALTER TABLE [dbo].[FactProperty]  WITH CHECK ADD  CONSTRAINT [FK__FactPrope__Prope__02FC7413] FOREIGN KEY([PropertyTypeID])
REFERENCES [dbo].[DimPropertyType] ([PropertyTypeID])
GO
ALTER TABLE [dbo].[FactProperty] CHECK CONSTRAINT [FK__FactPrope__Prope__02FC7413]
GO
ALTER TABLE [dbo].[FactProperty]  WITH CHECK ADD  CONSTRAINT [FK__FactPrope__RoofI__08B54D69] FOREIGN KEY([RoofID])
REFERENCES [dbo].[DimRoof] ([RoofID])
GO
ALTER TABLE [dbo].[FactProperty] CHECK CONSTRAINT [FK__FactPrope__RoofI__08B54D69]
GO
ALTER TABLE [dbo].[FactProperty]  WITH CHECK ADD  CONSTRAINT [FK__FactPrope__Stree__04E4BC85] FOREIGN KEY([StreetAccessID])
REFERENCES [dbo].[DimStreetAccess] ([StreetAccessID])
GO
ALTER TABLE [dbo].[FactProperty] CHECK CONSTRAINT [FK__FactPrope__Stree__04E4BC85]
GO
ALTER TABLE [dbo].[FactProperty]  WITH CHECK ADD  CONSTRAINT [FK__FactPrope__Utili__06CD04F7] FOREIGN KEY([UtilitiesID])
REFERENCES [dbo].[DimUtilities] ([UtilitiesID])
GO
ALTER TABLE [dbo].[FactProperty] CHECK CONSTRAINT [FK__FactPrope__Utili__06CD04F7]
GO
ALTER TABLE [dbo].[FactProperty]  WITH CHECK ADD  CONSTRAINT [FK_FactProperty_Property] FOREIGN KEY([PropertyID])
REFERENCES [dbo].[Property] ([PropertyID])
GO
ALTER TABLE [dbo].[FactProperty] CHECK CONSTRAINT [FK_FactProperty_Property]
GO
ALTER TABLE [dbo].[FactSale]  WITH CHECK ADD  CONSTRAINT [FK_FactSale_Property] FOREIGN KEY([PropertyID])
REFERENCES [dbo].[Property] ([PropertyID])
GO
ALTER TABLE [dbo].[FactSale] CHECK CONSTRAINT [FK_FactSale_Property]
GO
USE [master]
GO
ALTER DATABASE [HouseSales] SET  READ_WRITE 
GO
